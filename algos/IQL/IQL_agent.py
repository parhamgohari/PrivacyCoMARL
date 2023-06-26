import torch
from utils.network import Q_network, orthogonal_init
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from base.agent_base import Agent_Base, BaseGradSampleModule

class IQL_Agent(Agent_Base):
    def __init__(self, args, id, seed):
        super(IQL_Agent, self).__init__(args, id, seed)
        
        self.q_network = Q_network(args, self.input_dim)
        self.target_q_network = Q_network(args, self.input_dim)

        if self.use_anchoring:
            self.anchor_q_network = Q_network(args, self.input_dim)

        if self.use_orthogonal_init:
            orthogonal_init(self.q_network)

        self.eval_parameters = list(self.q_network.parameters())

        if self.use_RMS:
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        elif self.use_Adam:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)
        elif self.use_SGD:
            self.optimizer = torch.optim.SGD(self.eval_parameters, lr=self.lr, momentum=0.9, weight_decay=1e-2)
        else:
            raise NotImplementedError
        
        if self.use_dp:
            self.sample_rate = self.batch_size / self.buffer_size
            self.accountant = RDPAccountant()
            self.optimizer = DPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.grad_clip_norm,
                expected_batch_size=self.batch_size,
                )
            
            self.q_network = BaseGradSampleModule(self.q_network)
            self.target_q_network = BaseGradSampleModule(self.target_q_network)

            if self.use_anchoring:
                self.anchor_q_network = BaseGradSampleModule(self.anchor_q_network)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        if self.use_anchoring:
            self.update_anchor_params()

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        
        if self.use_anchoring:
            self.anchor_q_network.to(self.device)

    def train(self, total_steps):
        self.batch, self.max_episode_len, batch_length = self.replay_buffer.sample(self.seed)
        if self.use_poisson_sampling and batch_length is not None:
            self.current_batch_size = batch_length
        elif not self.use_poisson_sampling:
            self.current_batch_size = self.batch_size
        elif batch_length is None:
            return None
        
        if self.use_rnn:
            self.reset()
        
        self.empty_batch = False
        self.inputs = self.get_inputs(self.batch).clone().detach() # dimension: batch_size * max_episode_len * input_dim
        self.q_evals = self.running_models['q_network'](self.inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,action_dim)
        self.q_targets = self.target_models['q_network'](self.inputs)[:, 1:]
        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.running_models['q_network'](self.inputs[:, -1].unsqueeze(1)).reshape(self.current_batch_size, 1, -1)
                q_evals_next = torch.cat([self.q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,action_dim)
                q_evals_next[self.batch['avail_a'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, 1)
                self.q_targets = torch.gather(self.q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len)
            else:
                self.q_targets[self.batch['avail_a'][:, 1:] == 0] = -999999
                self.q_targets = self.q_targets.max(dim=-1)[0]
        
        targets = self.batch['r'] + self.gamma * self.q_targets.unsqueeze(-1) * (1 - self.batch['dw'])  # targets.shape=(batch_size, max_episode_len)
        td_error = self.q_evals.gather(dim=-1, index=self.batch['a'].unsqueeze(-1)) - targets.detach()
        mask_td_error = td_error * self.batch['active']
        loss = (mask_td_error ** 2).sum() / self.batch['active'].sum()

        self.optimizer.zero_grad()
        loss.backward()

        if self.use_grad_clip and not self.use_dp:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_clip_norm).item()
        self.update_seed()
        if torch.rand(1).item() < 1/self.buffer_throughput:
            self.optimizer.step()
        self.train_step += 1
        if self.use_dp and self.train_step <= self.buffer_size:
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)

        self.update_target_networks()

        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        verbose = {
            'id': self.id,
            'train_step': self.train_step,
            'loss': loss.item(),
            'grad_norm': grad_norm if self.use_grad_clip else None
        }
    
        return verbose
    
    def reset(self):
        if self.use_rnn:
            if self.use_dp:
                self.q_network.module.encoder.hidden = None
                self.target_q_network.module.encoder.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.module.encoder.hidden = None
            else:
                self.q_network.encoder.hidden = None
                self.target_q_network.encoder.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.encoder.hidden = None
        self.last_onehot_a = torch.zeros((self.action_dim), device=self.device)

    def update_target_networks(self):
        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                if self.use_anchoring:
                    self.anchor_q_network.load_state_dict(self.q_network.state_dict())
        else:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




        