import torch
from network import *
from buffer import ReplayBuffer
import numpy as np
from opacus import GradSampleModule
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.optimizers import DPOptimizer
from opacus.accountants import RDPAccountant



class QTRAN_Base_Agent(object):
    def __init__(self, args, id, seed):
        args.id = id
        self.id = id
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_agents = args.n_agents
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.add_last_action = args.add_last_action
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lambda_opt = args.lambda_opt
        self.lambda_nopt = args.lambda_nopt
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
        self.use_Adam = args.use_Adam
        self.use_lr_decay = args.use_lr_decay
        self.use_secret_sharing = args.use_secret_sharing
        self.use_poisson_sampling = args.use_poisson_sampling
        self.use_dp = args.use_dp
        self.noise_multiplier = args.noise_multiplier
        self.device = args.device
        self.replay_buffer = ReplayBuffer(args)
        self.buffer_throughput = args.buffer_throughput
        self.use_anchoring = args.use_anchoring
        self.anchoring_weight = 0

        self.last_onehot_a = torch.zeros(self.action_dim)

        self.update_seed(seed)

        # assert that if use_dp, then use_grad_clip and use_poission_sampling and noise_multiplier>0
        if self.use_dp:
            raise NotImplementedError
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
            if self.use_dp:
                raise NotImplementedError
            self.q_network = Q_network_RNN(args, self.input_dim + self.action_dim)
            self.target_q_network = Q_network_RNN(args, self.input_dim + self.action_dim)
            self.v_network = V_network_RNN(args, self.input_dim)
            self.q_jt_network = Q_jt_network_MLP(args, self.rnn_hidden_dim, self.seed)
            self.target_q_jt_network = Q_jt_network_MLP(args, self.rnn_hidden_dim, self.seed)
            self.update_seed()
            self.v_jt_network = V_jt_network_MLP(args, self.rnn_hidden_dim, self.seed)
            self.update_seed()
            
            if self.use_anchoring:
                raise NotImplementedError
        else:
            raise NotImplementedError
            self.q_network = Q_network_MLP(args, self.input_dim)
            self.target_q_network = Q_network_MLP(args, self.input_dim)
            self.h_v_network = H_V_network(args, self.input_dim)
            self.target_h_v_network = H_V_network(args, self.input_dim)
            self.q_jt_network = Q_jt_network_MLP(args, 2 * self.rnn_hidden_dim)
            self.target_q_jt_network = Q_jt_network_MLP(args, 2 * self.rnn_hidden_dim)
            self.v_jt_network = V_jt_network_MLP(args, self.rnn_hidden_dim)

            if self.use_anchoring:
                raise NotImplementedError

            

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_jt_network.load_state_dict(self.q_jt_network.state_dict())
        if self.use_anchoring:
            raise NotImplementedError
            self.anchor_q_network.load_state_dict(self.q_network.state_dict())

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.v_network.to(self.device)
        self.q_jt_network.to(self.device)
        self.target_q_jt_network.to(self.device)
        self.v_jt_network.to(self.device)
        if self.use_anchoring:
            raise NotImplementedError
            self.anchor_q_network.to(self.device)

        self.eval_parameters = list(self.q_network.parameters()) + list(self.q_jt_network.parameters()) + list(self.v_jt_network.parameters()) + list(self.v_network.parameters())
        if self.use_RMS:
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr, alpha = 0.99, eps = 0.00001)
        elif self.use_Adam:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr, weight_decay=1e-2)
        else: 
            self.optimizer = torch.optim.SGD(self.eval_parameters, lr=self.lr, weight_decay=1e-2)

        if self.use_dp:
            raise NotImplementedError
            self.q_network = GradSampleModule(self.q_network)
            self.q_network.register_full_backward_hook(forbid_accumulation_hook)
            self.target_q_network = GradSampleModule(self.target_q_network)
            self.target_q_network.register_full_backward_hook(forbid_accumulation_hook)
            self.v_network = GradSampleModule(self.h_v_network)
            self.v_network.register_full_backward_hook(forbid_accumulation_hook)
            self.q_jt_network = GradSampleModule(self.q_jt_network)
            self.q_jt_network.register_full_backward_hook(forbid_accumulation_hook)
            self.target_q_jt_network = GradSampleModule(self.target_q_jt_network)
            self.target_q_jt_network.register_full_backward_hook(forbid_accumulation_hook)
            self.v_jt_network = GradSampleModule(self.v_jt_network)
            self.v_jt_network.register_full_backward_hook(forbid_accumulation_hook)
            if self.use_anchoring:
                raise NotImplementedError
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

    def update_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            self.seed = seed
        else:
            torch.manual_seed(self.seed)
            self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    def choose_action(self, local_obs, avail_a, epsilon, seed):
        self.obs = local_obs
        self.avail_a = avail_a
        with torch.no_grad():
            self.update_seed(seed)
            if torch.rand(1).item() <= epsilon:
                a = torch.where(torch.tensor(avail_a)==1)[0][torch.randint(0, sum(avail_a), (1,)).item()].item()

            else:
                q_logits = torch.zeros((self.action_dim,))
                rnn_hidden_state = self.q_network.rnn_hidden
                for a in range(self.action_dim):
                    inputs = []
                    obs = torch.tensor(local_obs, dtype=torch.float32, device=self.device)
                    inputs.append(obs)
                    if self.add_last_action:
                        inputs.append(self.last_onehot_a)
                    inputs.append(torch.eye(self.action_dim)[a])
                    inputs = torch.cat(inputs, dim=-1)
                    q_logits[a] = self.q_network(inputs).item()
                    self.q_network.rnn_hidden = rnn_hidden_state   
                avail_a = torch.tensor(avail_a, dtype=torch.bool, device="cpu")
                q_logits[avail_a == False] = float('-inf')
                a = q_logits.max(dim=-1)[1].item()
        self.current_a = a
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
        if len(shares.shape) == 3:
            shares = shares.permute(1, 2, 0)
        elif len(shares.shape) == 4:
            shares = shares.permute(1, 2, 3, 0)
        return shares
    
    def decrypt(self, shares):
        return self.decoder(torch.sum(shares, dim=0) % self.Q)
    
    def initiate_peer2peer_messaging(self, seed):

        # Draw a batch from the replay buffer and skip if the batch is empty
        self.batch, self.max_episode_len, batch_length = self.replay_buffer.sample(seed)
        self.current_batch_size = batch_length
        if batch_length is None:
            self.empty_batch = True
            return
        
        self.empty_batch = False
        
        # Get the inputs
        self.inputs = self.get_inputs(self.batch).clone().detach()
        
        if self.use_rnn:
            self.q_network.rnn_hidden = None
            self.target_q_network.rnn_hidden = None
            self.v_network.rnn_hidden = None

            q_evals, q_targets, v_evals, h_q_evals, h_q_targets, h_v_evals = [], [], [], [], [], []
            for t in range(self.max_episode_len):  # t=0,1,2,...(episode_len-1)
                q_network_rnn_state = self.q_network.rnn_hidden
                target_q_network_rnn_state = self.target_q_network.rnn_hidden
                q_eval_logits, q_target_logits, h_q_logits, h_q_target_logits = [], [], [], []
                for a in range(self.action_dim):
                    one_hot_a = torch.eye(self.action_dim)[a]
                    
                    inputs_w_one_hot_a = torch.cat([self.inputs[:, t], one_hot_a.unsqueeze(0).repeat(self.inputs.shape[0], 1).clone()], dim=-1)
                    q_eval_logits.append(self.q_network(inputs_w_one_hot_a).squeeze(-1))
                    h_q_logits.append(self.q_network.rnn_hidden)
                    self.q_network.rnn_hidden = q_network_rnn_state

                    next_input_w_onehot_a = torch.cat([self.inputs[:, t + 1], one_hot_a.unsqueeze(0).repeat(self.inputs.shape[0], 1).clone()], dim=-1)
                    q_target_logits.append(self.target_q_network(next_input_w_onehot_a).squeeze(-1))
                    h_q_target_logits.append(self.target_q_network.rnn_hidden)
                    self.target_q_network.rnn_hidden = target_q_network_rnn_state
                
                q_eval_logits_stacked = torch.stack(q_eval_logits, dim=-1)
                q_target_logits_stacked = torch.stack(q_target_logits, dim=-1)
                h_q_logits_stacked = torch.stack(h_q_logits, dim=-1)
                h_q_target_logits_stacked = torch.stack(h_q_target_logits, dim=-1)
                
                v_eval = self.v_network(self.inputs[:, t + 1].reshape(-1, self.input_dim)).squeeze(-1)
                h_v_eval = self.v_network.rnn_hidden
                v_evals.append(v_eval.reshape(self.batch_size, -1))
                h_v_evals.append(h_v_eval)

                q_evals.append(q_eval_logits_stacked)
                q_targets.append(q_target_logits_stacked)
                h_q_evals.append(h_q_logits_stacked)
                h_q_targets.append(h_q_target_logits_stacked)
            
            self.q_evals = torch.stack(q_evals, dim=1)
            self.q_targets = torch.stack(q_targets, dim=1)
            self.v_evals = torch.stack(v_evals, dim=1)
            self.h_q_evals = torch.stack(h_q_evals, dim=1)
            self.h_q_targets = torch.stack(h_q_targets, dim=1)
            self.h_v_evals = torch.stack(h_v_evals, dim=1)

        else:
            raise NotImplementedError
            # self.q_evals = self.q_network(self.inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,action_dim)
            # self.q_targets = self.target_q_network(self.inputs[:, 1:])
        
        # Mask the unavailable actions
        q_evals_masked = self.q_evals.clone()
        q_targets_masked = self.q_targets.clone()
        q_evals_masked[self.batch['avail_a'][:, :-1] == 0] = float('-inf')
        q_targets_masked[self.batch['avail_a'][:, 1:] == 0] = float('-inf')
        self.a_opt = q_evals_masked.max(dim=-1)[1]  # a_opt is the actions that maximize the agent's q value

        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions, and use target_net to compute q_target
                self.a_opt_next = q_evals_masked.max(dim=-1)[1]  # q_evals.shape=(batch_size, max_episode_len)
            else:
                self.a_opt_next = q_targets_masked.max(dim=-1)[1]


    def peer2peer_messaging_for_computing_q_jt_sum(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
              # q_targets.shape=(batch_size, max_episode_len)
            self.message_to_send = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size, self.max_episode_len, self.n_agents), device=self.device, dtype=torch.float32)

            self.contribution_to_q_jt_sum = torch.gather(self.q_evals, dim=-1, index=self.batch['a'].unsqueeze(-1)).squeeze(-1)
            

            # make n_agent copies of q_evals and q_targets each of which multiplied by 1/n_agents
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.contribution_to_q_jt_sum
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.contribution_to_q_jt_sum
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, self.n_agents) / self.n_agents # q_evals.shape=(batch_size, max_episode_len, n_agents, n_agents)
    

        elif '2' in mode:
            self.message_to_rece[:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2)  # sum_q_vals.shape=(batch_size, max_episode_len)
            return self.sum_shares
        
        elif '4' in mode:
            self.q_jt_sum = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            assert self.q_jt_sum.requires_grad == False
            assert self.contribution_to_q_jt_sum.requires_grad == True
            self.q_jt_sum +=  self.contribution_to_q_jt_sum - self.contribution_to_q_jt_sum.detach()
            assert self.q_jt_sum.requires_grad == True

    def peer2peer_messaging_for_computing_q_jt_sum_opt(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.contribution_to_q_jt_sum_opt = torch.gather(self.q_evals, dim=-1, index=self.a_opt.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.contribution_to_q_jt_sum_opt
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.contribution_to_q_jt_sum_opt
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, self.n_agents) / self.n_agents # q_evals.shape=(batch_size, max_episode_len, n_agents, n_agents)
        elif '2' in mode:
            self.message_to_rece[:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=2)  # sum_q_vals.shape=(batch_size, max_episode_len)
            return self.sum_shares
        
        elif '4' in mode:
            self.q_jt_sum_opt = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            assert self.q_jt_sum_opt.requires_grad == False
            assert self.contribution_to_q_jt_sum_opt.requires_grad == True
            self.q_jt_sum_opt += self.contribution_to_q_jt_sum_opt - self.contribution_to_q_jt_sum_opt.detach()
            assert self.q_jt_sum_opt.requires_grad == True

    def peer2peer_messaging_for_computing_q_jt(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.message_to_send = torch.zeros((self.current_batch_size, self.max_episode_len, self.rnn_hidden_dim, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size, self.max_episode_len, self.rnn_hidden_dim, self.n_agents), device=self.device, dtype=torch.float32)
            reshaped_a = self.batch['a'].unsqueeze(-1).expand(-1, -1, self.h_q_evals.size(-2)).unsqueeze(-1)
            self.contribution_to_sum_h_q_evals = torch.gather(self.h_q_evals, dim=-1, index=reshaped_a).squeeze(-1)
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.contribution_to_sum_h_q_evals
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.contribution_to_sum_h_q_evals
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents
                    
        elif '2' in mode:
            self.message_to_rece[:,:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)
            return self.sum_shares
        
        elif '4' in mode:
            self.sum_h_q_evals = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            # self.sum_h_q_evals_excluding_self = self.sum_h_evals[:,:,:self.rnn_hidden_dim] - self.h_q_evals.detach()
            # assert self.sum_h_q_evals_excluding_self.requires_grad == False
            # assert self.h_v_evals.requires_grad == True
            # self.q_jt_evals = self.q_jt_network(torch.concat([self.sum_h_q_evals_excluding_self, self.h_v_evals], dim=-1))
            # assert self.q_jt_evals.requires_grad == True
            assert self.sum_h_q_evals.requires_grad == False
            self.sum_h_q_evals += self.contribution_to_sum_h_q_evals - self.contribution_to_sum_h_q_evals.detach()
            assert self.sum_h_q_evals.requires_grad == True
            self.q_jt = self.q_jt_network(self.sum_h_q_evals).squeeze(-1)
            assert self.q_jt.requires_grad == True
            


    def peer2peer_messaging_for_computing_y_dqn(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.reshaped_a_opt = self.a_opt.unsqueeze(-1).expand(-1, -1, self.h_q_targets.size(-2)).unsqueeze(-1)
            self.contribution_to_sum_h_q_targets = torch.gather(self.h_q_targets, dim=-1, index=self.reshaped_a_opt).squeeze(-1)
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.contribution_to_sum_h_q_targets
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.contribution_to_sum_h_q_targets
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents
                    
        elif '2' in mode:
            self.message_to_rece[:,:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)
            return self.sum_shares
        
        elif '4' in mode:
            self.sum_h_q_targets = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            # self.sum_h_q_targets_excluding_self = self.sum_h_targets[:,:,:self.rnn_hidden_dim] - self.h_q_targets.detach()
            # assert self.sum_h_q_targets_excluding_self.requires_grad == False
            # with torch.no_grad():
            #     self.q_jt_targets = self.target_q_jt_network(torch.concat([self.sum_h_q_targets_excluding_self, self.h_v_targets], dim=-1))
            self.sum_h_q_targets += self.contribution_to_sum_h_q_targets - self.contribution_to_sum_h_q_targets.detach()
            assert self.sum_h_q_targets.requires_grad == True
            with torch.no_grad():
                self.q_jt_targets = self.q_jt_network(self.sum_h_q_targets).squeeze(-1)
            assert self.q_jt_targets.requires_grad == False
            self.y_dqn = self.batch['r'].squeeze(-1) + self.gamma * (1 - self.batch['dw'].squeeze(-1)) * self.q_jt_targets
            assert self.y_dqn.requires_grad == False

    def peer2peer_messaging_for_computing_q_jt_opt(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.contribution_to_sum_h_q_evals_opt = torch.gather(self.h_q_evals, dim=-1, index=self.reshaped_a_opt).squeeze(-1)
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.contribution_to_sum_h_q_evals_opt
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.contribution_to_sum_h_q_evals_opt
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents

        elif '2' in mode:
            self.message_to_rece[:,:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)
            return self.sum_shares
        
        elif '4' in mode:
            self.sum_h_q_evals_opt = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            assert self.sum_h_q_evals_opt.requires_grad == False
            assert self.contribution_to_sum_h_q_evals_opt.requires_grad == True
            self.sum_h_q_evals_opt += self.contribution_to_sum_h_q_evals_opt - self.contribution_to_sum_h_q_evals_opt.detach()
            assert self.sum_h_q_evals_opt.requires_grad == True
            self.q_jt_opt = self.q_jt_network(self.sum_h_q_evals_opt).squeeze(-1)
            assert self.q_jt_opt.requires_grad == True

    def peer2peer_messaging_for_computing_v_jt(self, seed, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            with torch.no_grad():
                if self.use_secret_sharing:
                    self.update_seed(seed)
                    self.secret = self.h_v_evals
                    self.secret_shares = self.encrypt(self.secret)
                else:
                    self.secret = self.h_v_evals
                    self.secret_shares = self.secret.unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents
        
        elif '2' in mode:
            self.message_to_rece[:,:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)
            return self.sum_shares
        
        elif '4' in mode:
            self.sum_h_v_evals = self.decrypt(sender_message) if self.use_secret_sharing else sender_message
            assert self.sum_h_v_evals.requires_grad == False
            assert self.h_v_evals.requires_grad == True
            self.sum_h_v_evals += self.h_v_evals - self.h_v_evals.detach()
            assert self.sum_h_v_evals.requires_grad == True
            self.v_jt = self.v_jt_network(self.sum_h_v_evals).squeeze(-1)
            assert self.v_jt.requires_grad == True

            

    def train(self, total_steps):

        self.l_td = self.q_jt - self.y_dqn.detach()
        self.l_opt = self.q_jt_sum_opt - self.q_jt_opt.detach() + self.v_jt
        self.l_nopt = torch.clamp(self.q_jt_sum - self.q_jt.detach() + self.v_jt, max=0)
        self.l_td_v = self.contribution_to_q_jt_sum.detach() - self.batch['r'].squeeze(-1) - self.gamma * (1 - self.batch['dw'].squeeze(-1)) * self.v_evals.squeeze(-1)

        self.train_step += 1

        masked_l_td = self.l_td * self.batch['active'].squeeze(-1)
        l_td = (masked_l_td ** 2).sum() / self.batch['active'].sum()

        masked_l_opt = self.l_opt * self.batch['active'].squeeze(-1)
        l_opt = (masked_l_opt ** 2).sum() / self.batch['active'].sum()

        masked_l_nopt = self.l_nopt * self.batch['active'].squeeze(-1)
        l_nopt = (masked_l_nopt ** 2).sum() / self.batch['active'].sum()

        masked_l_td_v = self.l_td_v * self.batch['active'].squeeze(-1)
        l_td_v = (masked_l_td_v ** 2).sum() / self.batch['active'].sum()


        loss = (l_td + l_td_v + self.lambda_opt * l_opt + self.lambda_nopt * l_nopt)



        if self.use_anchoring:
            raise NotImplementedError
            # compute the the 2-norm difference between the anchor_q_network and the q_network and add it to the loss
            loss += self.anchoring_weight * self.anchoring_loss(self.q_network, self.anchor_q_network)


        self.optimizer.zero_grad()
        loss.backward()


        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_clip_norm)
        
        if np.random.rand() < 1/self.buffer_throughput:
            self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                self.target_q_jt_network.load_state_dict(self.q_jt_network.state_dict())
                


        else:
            # Softly update the target networks
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.q_jt_network.parameters(), self.target_q_jt_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        verbose = {
            'id': self.id,
            'train_step': self.train_step,
            'loss': loss.item(),
            'l_td': l_td.item(),
            'l_opt': l_opt.item(),
            'l_nopt': l_nopt.item()
        }
    
        return verbose

        

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
    
    def store_transition(self, episode_step, r, dw):
        self.replay_buffer.store_transition(
            episode_step=episode_step,
            obs = self.obs,
            avail_a=self.avail_a,
            last_onehot_a=self.last_onehot_a,
            a=self.current_a,
            r=r,
            dw=dw
        )
        self.last_onehot_a = torch.zeros(self.action_dim)
        self.last_onehot_a[self.current_a] = 1
    
    def store_last_step(self, episode_step, obs, avail_a):
        self.replay_buffer.store_last_step(
            episode_step=episode_step,
            obs=obs,
            avail_a=avail_a,
            last_onehot_a=self.last_onehot_a
        )
        self.last_onehot_a = torch.zeros(self.action_dim)



            
                    
                    
        
        