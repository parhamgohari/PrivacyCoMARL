import torch
from network import Q_network_RNN, Q_network_MLP
from buffer import ReplayBuffer
import numpy as np
from opacus import GradSampleModule
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.optimizers import DPOptimizer
from opacus.accountants import RDPAccountant



class VDN_Agent(object):
    def __init__(self, args, id, seed):
        args.id = id
        self.id = id
        self.n_agents = args.n_agents
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.add_last_action = args.add_last_action
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.grad_clip_norm = args.grad_clip_norm
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size  
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.use_secret_sharing = args.use_secret_sharing
        self.use_poisson_sampling = args.use_poisson_sampling
        self.use_dp = args.use_dp
        self.noise_multiplier = args.noise_multiplier
        self.seed = seed
        self.device = args.device
        self.replay_buffer = ReplayBuffer(args)
        self.buffer_throughput = args.buffer_throughput
        self.use_anchoring = args.use_anchoring
        self.anchoring_weight = 0

        # assert that if use_dp, then use_grad_clip and use_poission_sampling and noise_multiplier>0
        if self.use_dp:
            assert self.use_grad_clip
            assert self.use_poisson_sampling
            assert self.noise_multiplier > 0

        self.base = 10
        self.precision = 5
        self.Q = 2**31 - 1

        self.input_dim = self.obs_dim
        if self.add_last_action:
            self.input_dim += self.action_dim

        if self.use_rnn:
            self.q_network = Q_network_RNN(args, self.input_dim)
            self.target_q_network = Q_network_RNN(args, self.input_dim)
            if self.use_anchoring:
                self.anchor_q_network = Q_network_RNN(args, self.input_dim)

        else:
            self.q_network = Q_network_MLP(args, self.input_dim)
            self.target_q_network = Q_network_MLP(args, self.input_dim)
            if self.use_anchoring:
                self.anchor_q_network = Q_network_MLP(args, self.input_dim)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        if self.use_anchoring:
            self.anchor_q_network.load_state_dict(self.q_network.state_dict())

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        if self.use_anchoring:
            self.anchor_q_network.to(self.device)

        self.eval_parameters = list(self.q_network.parameters())
        if self.use_RMS:
            self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=self.lr)
        else:
            # self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
            self.optimizer = torch.optim.SGD(self.q_network.parameters(), lr=self.lr, weight_decay=1e-2, momentum=0.9)

        if self.use_dp:
            self.q_network = GradSampleModule(self.q_network)
            self.q_network.register_full_backward_hook(forbid_accumulation_hook)
            self.target_q_network = GradSampleModule(self.target_q_network)
            self.target_q_network.register_full_backward_hook(forbid_accumulation_hook)
            if self.use_anchoring:
                self.anchor_q_network = GradSampleModule(self.anchor_q_network)
                self.anchor_q_network.register_full_backward_hook(forbid_accumulation_hook)
            self.sample_rate = self.batch_size / self.buffer_size
            self.sampling_method = 'poisson'
            self.optimizer = DPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.grad_clip_norm,
                expected_batch_size=self.batch_size,
                ) 
            self.accountant = RDPAccountant()
            self.optimizer.attach_step_hook(self.accountant.get_optimizer_hook_fn(sample_rate=self.sample_rate))

        self.train_step = 0

    def choose_action(self, local_obs, last_onehot_a, avail_a, epsilon, evaluate_anchor_q=False):
        with torch.no_grad():
            if np.random.uniform() < epsilon:
                return np.random.choice(np.nonzero(avail_a)[0])
            else:
                inputs = []
                obs = torch.tensor(local_obs, dtype=torch.float32, device=self.device)
                inputs.append(obs)
                if self.add_last_action:
                    last_a = torch.tensor(last_onehot_a, dtype=torch.float32, device=self.device)
                    inputs.append(last_a)
                
                # concatenate all inputs
                inputs = torch.cat(inputs, dim=0)

                if evaluate_anchor_q:
                    assert self.use_anchoring
                    q_values = self.anchor_q_network(inputs)
                else:
                    q_values = self.q_network(inputs)
                avail_a = torch.tensor(avail_a, dtype=torch.float32, device="cpu")
                q_values[avail_a == 0.0] = -float('inf')
                a = torch.max(q_values, dim=0)[1].item()
            return a
    
    # Fixed point encoding
    def encoder(self, x):
        encoding = (self.base**self.precision * x % self.Q).clone().detach().int()
        return encoding

    # Fixed point decoding
    def decoder(self, x):
        x[x > self.Q/2] = x[x > self.Q/2] - self.Q
        return x / self.base**self.precision
    
    # Additive secret sharing
    def encrypt(self, x):
        shares = []
        for i in range(self.n_agents-1): 
            shares.append(torch.randint(0, self.Q, x.shape, device=self.device))
        shares.append(self.Q - sum(shares) % self.Q + self.encoder(x))
        shares = torch.stack(shares, dim=0)
        # swap the first dimension with the last dimension
        shares = shares.permute(1, 2, 0)
        return shares
    
    def decrypt(self, shares):
        return self.decoder(torch.sum(shares, dim=0) % self.Q)


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
            #self.message_to_send = empty tensor with shape (batch_size, max_episode_len, n_agents)
            self.message_to_send = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)
            self.train_step += 1
            if self.use_rnn:
                self.q_network.rnn_hidden = torch.zeros(64, device=self.device)
                self.target_q_network.rnn_hidden = torch.zeros(64, device=self.device)
        elif '1' in mode:
            if self.use_rnn:
                self.q_evals, self.q_targets = [], []
                for t in range(self.max_episode_len):  # t=0,1,2,...(episode_len-1)
                    if self.use_dp:
                        self.q_eval = []
                        self.q_target = []
                        if self.use_double_q:
                            q_eval_last = []
                        for sample in range(self.current_batch_size):
                            self.q_eval.append(self.q_network(self.inputs[sample, t, :].reshape(self.input_dim)))  
                            self.q_target.append(self.target_q_network(self.inputs[sample, t + 1, :].reshape(self.input_dim)))
                            if t == self.max_episode_len - 1 and self.use_double_q:
                                q_eval_last.append(self.q_network(self.inputs[sample, t + 1, :].reshape(self.input_dim)))
        
                        self.q_eval = torch.stack(self.q_eval, dim=0)
                        self.q_target = torch.stack(self.q_target, dim=0)

                    else:
                        self.q_eval = self.q_network(self.inputs[:, t].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size,action_dim)
                        self.q_target = self.target_q_network(self.inputs[:, t + 1].reshape(-1, self.input_dim))
                    self.q_evals.append(self.q_eval.reshape(self.current_batch_size, -1))  # q_eval.shape=(batch_size,action_dim)
                    self.q_targets.append(self.q_target.reshape(self.current_batch_size, -1))

                # Stack them according to the time (dim=1)
                self.q_evals = torch.stack(self.q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,action_dim)
                self.q_targets = torch.stack(self.q_targets, dim=1)
            else:
                self.q_evals = self.q_network(self.inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,action_dim)
                self.q_targets = self.target_q_network(self.inputs[:, 1:])

            with torch.no_grad():
                if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                    if not self.use_rnn or not self.use_dp:
                        q_eval_last = self.q_network(self.inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.current_batch_size, 1, -1)
                    else:
                        q_eval_last = torch.stack(q_eval_last, dim=0).reshape(self.current_batch_size, 1, -1)
                    q_evals_next = torch.cat([self.q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,action_dim)
                    q_evals_next[self.batch['avail_a'][:, 1:] == 0] = -999999
                    a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, 1)
                    self.q_targets = torch.gather(self.q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len)
                else:
                    self.q_targets[self.batch['avail_a'][:, 1:] == 0] = -999999
                    self.q_targets = self.q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len)

            
            self.q_evals = torch.gather(self.q_evals, dim=-1, index=self.batch['a'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len)
            # make n_agent copies of q_evals and q_targets each of which multiplied by 1/n_agents
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.secret = self.q_evals - ((1.0/self.n_agents) * self.batch['r'].squeeze(-1) + self.gamma * (1-self.batch['dw'].squeeze(-1)) * self.q_targets)
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.q_evals - ((1.0/self.n_agents) * self.batch['r'].squeeze(-1) + self.gamma * (1-self.batch['dw'].squeeze(-1)) * self.q_targets)
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, self.n_agents) / self.n_agents # q_evals.shape=(batch_size, max_episode_len, n_agents, n_agents)
    

        elif '2' in mode:
            self.message_to_rece[:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2)  # sum_q_vals.shape=(batch_size, max_episode_len)
            return self.sum_shares

            
            

    def train(self, total_steps, sum_shares):
        td_error = sum_shares - self.q_evals.detach()
        td_error += self.q_evals
        mask_td_error = td_error * self.batch['active'].squeeze(-1)

        loss = (mask_td_error ** 2).sum() / self.batch['active'].sum()

        if self.use_anchoring:
            # compute the the 2-norm difference between the anchor_q_network and the q_network and add it to the loss
            loss += self.anchoring_weight * self.anchoring_loss(self.q_network, self.anchor_q_network)


        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        
        if np.random.rand() < 1/self.buffer_throughput:
            self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def anchoring_loss(self, q_network, anchor_q_network):
        loss = 0
        for param, anchor_param in zip(q_network.parameters(), anchor_q_network.parameters()):
            loss += torch.norm(param - anchor_param.detach())
        return loss
    
    def lr_decay(self, total_steps):  # Learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now


    
    def get_inputs(self, batch):
        inputs = []
        obs = batch['obs'].clone().detach()
        inputs.append(obs)
        if self.add_last_action:
            last_a = batch['last_onehot_a'].clone().detach()
            inputs.append(last_a)
        inputs = torch.cat(inputs, dim=2)
        return inputs



            
                    
                    
        
        