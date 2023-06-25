from base.agent_base import Agent_Base
import torch

class IQL_Agent(Agent_Base):
    def __init__(self, args, id, seed):
        super(IQL_Agent, self).__init__(args, id, seed)
        self.init_optimizer()

    def train(self, total_steps):
        self.batch, self.max_episode_len, batch_length = self.replay_buffer.sample(self.seed)
        if self.use_poisson_sampling and batch_length is not None:
            self.current_batch_size = batch_length
        elif not self.use_poisson_sampling:
            self.current_batch_size = self.batch_size
        elif batch_length is None:
            return
        
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

        grad_norm = None
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

        return loss.item(), grad_norm




        