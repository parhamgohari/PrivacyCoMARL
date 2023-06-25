import torch
from utils.network import Q_network, orthogonal_init
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from base.agent_base import Agent_Base, BaseGradSampleModule

class VDN_Agent(Agent_Base):
    def __init__(self, args, id, seed):
        super(VDN_Agent, self).__init__(args, id, seed)

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


    def peer2peer_messaging(self, seed, mode, sender_id = None, sender_message = None):
        if '0' in mode:
            self.batch, self.max_episode_len, batch_length = self.replay_buffer.sample(seed)
            if self.use_poisson_sampling and batch_length is not None:
                self.current_batch_size = batch_length
            elif not self.use_poisson_sampling:
                self.current_batch_size = self.batch_size
            elif batch_length is None:
                self.empty_batch = True
                return
            self.empty_batch = False
            self.inputs = self.get_inputs(self.batch).clone().detach() # dimension: batch_size * max_episode_len * input_dim
            self.message_to_send = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)
            if self.use_rnn:
                self.reset()
        elif '1' in mode:
            self.q_evals = self.q_network(self.inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,action_dim)
            with torch.no_grad():
                self.q_targets = self.target_q_network(self.inputs)[:, 1:]
                if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                    q_eval_last = self.q_network(self.inputs[:, -1].unsqueeze(1)).reshape(self.current_batch_size, 1, -1)
                    q_evals_next = torch.cat([self.q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,action_dim)
                    q_evals_next[self.batch['avail_a'][:, 1:] == 0] = -999999
                    a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, 1)
                    self.q_targets = torch.gather(self.q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len)
                else:
                    self.q_targets[self.batch['avail_a'][:, 1:] == 0] = -999999
                    self.q_targets = self.q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len)
            self.q_evals = torch.gather(self.q_evals, dim=-1, index=self.batch['a'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len)
            self.secret = self.q_evals - ((1.0/self.n_agents) * self.batch['r'].squeeze(-1) + self.gamma * (1-self.batch['dw'].squeeze(-1)) * self.q_targets)
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed()
                    self.secret_shares = self.encrypt(self.secret.clone().detach())
                else:
                    self.secret_shares = self.secret.clone().detach().unsqueeze(-1).repeat(1, 1, self.n_agents) / self.n_agents # q_evals.shape=(batch_size, max_episode_len, n_agents, n_agents)
    

        elif '2' in mode:
            self.message_to_rece[:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)  # sum_q_vals.shape=(batch_size, max_episode_len)
            return self.sum_shares
        
        elif '4' in mode:
            self.td_error = (self.decrypt(sender_message) if self.use_secret_sharing else sender_message) - self.secret.detach() + self.secret


    def train(self, total_steps):
        mask_td_error = self.td_error * self.batch['active'].squeeze(-1)

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

            
                    
                    
        
        