import torch
from utils.network import Encoder, Q_network
from base.agent_base import Agent_Base

class QTRAN_Agent(Agent_Base):
    def __init__(self, args, id, seed):
        super(QTRAN_Agent, self).__init__(args, id, seed)
        self.q_network_hidden_dim = args.q_network_hidden_dim
        self.qtran_hidden_dim = args.qtran_hidden_dim
        self.vtran_hidden_dim = args.vtran_hidden_dim
        self.lambda_opt = args.lambda_opt
        self.lambda_nopt = args.lambda_nopt

        self.q_network = Q_network(args, self.input_dim)
        self.target_q_network = Q_network(args, self.input_dim)

        self.qtran_network = torch.nn.Sequential(
            torch.nn.Linear(self.q_network_hidden_dim + self.action_dim, self.qtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.qtran_hidden_dim, self.qtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.qtran_hidden_dim, 1)
        )
        self.target_qtran_network = torch.nn.Sequential(
            torch.nn.Linear(self.q_network_hidden_dim + self.action_dim, self.qtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.qtran_hidden_dim, self.qtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.qtran_hidden_dim, 1)
        )
        self.vtran_network = torch.nn.Sequential(
            torch.nn.Linear(self.q_network_hidden_dim, self.vtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.vtran_hidden_dim, self.vtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.vtran_hidden_dim, 1)
        )
        self.target_vtran_network = torch.nn.Sequential(
            torch.nn.Linear(self.q_network_hidden_dim, self.vtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.vtran_hidden_dim, self.vtran_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.vtran_hidden_dim, 1)
        )
        

        self.running_models = [self.q_network, self.qtran_network, self.vtran_network]
        self.target_models = [self.target_q_network, self.target_qtran_network, self.target_vtran_network]

        if self.use_anchoring:
            self.anchor_q_network = Q_network(args, self.input_dim)

        self.init_optimizer()

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
            self.reset()
        
        self.encoding = self.q_network.encoder(self.inputs[:, :-1])
        self.q_logits = self.q_network.fc(self.encoding)  # q_evals.shape=(batch_size,max_episode_len,action_dim)
        
        with torch.no_grad():
            target_encoding = self.target_q_network.encoder(self.inputs)
            self.q_target_logits = self.target_q_network.fc(target_encoding)
            self.q_targets = self.q_target_logits[:, 1:]
            self.target_encoding_next = target_encoding[:, 1:]
            self.a_opt = torch.argmax(self.q_logits, dim = -1)

            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                last_encoding = self.q_network.encoder(self.inputs[:, -1].unsqueeze(1)).reshape(self.current_batch_size, 1, -1)
                q_eval_last = self.q_network.fc(last_encoding).reshape(self.current_batch_size, 1, -1)
                q_evals_next = torch.cat([self.q_logits[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,action_dim)
                q_evals_next[self.batch['avail_a'][:, 1:] == 0] = -999999
                self.a_opt_next = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, 1)
                self.q_targets = torch.gather(self.q_targets, dim=-1, index=self.a_opt_next).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len)
            else:
                self.q_targets[self.batch['avail_a'][:, 1:] == 0] = -999999
                self.q_targets, self.a_opt_next = self.q_targets.max(dim=-1)  # q_targets.shape=(batch_size, max_episode_len)

        
        self.q_evals = torch.gather(self.q_logits, dim=-1, index=self.batch['a'].unsqueeze(-1)).squeeze(-1)
        self.q_evals_opt = torch.gather(self.q_logits, dim=-1, index=self.a_opt.unsqueeze(-1)).squeeze(-1)

        self.onehot_a = torch.eye(self.action_dim).expand(self.current_batch_size, self.max_episode_len, -1, -1).gather(dim = 3, index = self.batch['a'].view(self.current_batch_size, self.max_episode_len, 1, 1).repeat(1, 1, self.action_dim, self.action_dim))[:, :, :, 0].to(self.device)
        self.onehot_a_opt = torch.eye(self.action_dim).expand(self.current_batch_size, self.max_episode_len, -1, -1).gather(dim = 3, index = self.a_opt.view(self.current_batch_size, self.max_episode_len, 1, 1).repeat(1, 1, self.action_dim, self.action_dim))[:, :, :, 0].to(self.device)
        self.onehot_a_opt_next = torch.eye(self.action_dim).expand(self.current_batch_size, self.max_episode_len, -1, -1).gather(dim = 3, index = self.a_opt_next.view(self.current_batch_size, self.max_episode_len, 1, 1).repeat(1, 1, self.action_dim, self.action_dim))[:, :, :, 0].to(self.device)

        self.h_q = torch.cat([self.encoding, self.onehot_a], dim = -1)
        self.h_q_opt = torch.cat([self.encoding, self.onehot_a_opt], dim = -1)
        self.h_q_opt_next = torch.cat([self.target_encoding_next, self.onehot_a_opt_next], dim = -1).detach()
        self.h_v = self.encoding

    def peer2peer_messaging(self, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.secret = []
            
            self.secret.append(self.h_q)
            self.h_q_start = 0
            self.h_q_end = self.q_network_hidden_dim + self.action_dim
            
            self.secret.append(self.h_v)
            self.h_v_start = self.h_q_end
            self.h_v_end = self.h_v_start + self.q_network_hidden_dim
            
            self.secret.append(self.h_q_opt)
            self.h_q_opt_start = self.h_v_end
            self.h_q_opt_end = self.h_q_opt_start + self.q_network_hidden_dim + self.action_dim

            self.secret.append(self.h_q_opt_next)
            self.h_q_opt_next_start = self.h_q_opt_end
            self.h_q_opt_next_end = self.h_q_opt_next_start + self.q_network_hidden_dim + self.action_dim

            self.secret.append(self.q_evals.unsqueeze(-1))
            self.q_evals_start = self.h_q_opt_next_end
            self.q_evals_end = self.q_evals_start + 1

            self.secret.append(self.q_evals_opt.unsqueeze(-1))
            self.q_evals_opt_start = self.q_evals_end
            self.q_evals_opt_end = self.q_evals_opt_start + 1

            self.secret = torch.cat(self.secret, dim = -1)

            self.message_to_send = torch.zeros((self.current_batch_size, self.max_episode_len, self.secret.shape[-1], self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size, self.max_episode_len, self.secret.shape[-1], self.n_agents), device=self.device, dtype=torch.float32)

            if self.use_secret_sharing:
                self.update_seed()
                self.secret_shares = self.encrypt(self.secret.detach())
            else:
                self.secret_shares = self.secret.detach().unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents

        elif '2' in mode:
            self.message_to_rece[:,:,:,sender_id] = sender_message

        elif '3' in mode:
            if self.use_secret_sharing:
                self.update_seed()
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1) % self.Q
            else:
                self.sum_shares = torch.sum(self.message_to_rece, dim=-1)
            return self.sum_shares
        
        elif '4' in mode:
            self.sum_secrets = (self.decrypt(sender_message) if self.use_secret_sharing else sender_message) - self.secret.detach() + self.secret

    def train(self, total_steps):
        sum_h_q = self.sum_secrets[:, :, self.h_q_start:self.h_q_end]
        sum_h_v = self.sum_secrets[:, :, self.h_v_start:self.h_v_end]
        sum_h_q_opt = self.sum_secrets[:, :, self.h_q_opt_start:self.h_q_opt_end]
        sum_h_q_opt_next = self.sum_secrets[:, :, self.h_q_opt_next_start:self.h_q_opt_next_end]
        sum_q_evals = self.sum_secrets[:, :, self.q_evals_start:self.q_evals_end]
        sum_q_evals_opt = self.sum_secrets[:, :, self.q_evals_opt_start:self.q_evals_opt_end]

        qtran = self.qtran_network(sum_h_q)
        qtran_opt = self.qtran_network(sum_h_q_opt)
        qtran_opt_next = self.target_qtran_network(sum_h_q_opt_next).detach()
        vtran = self.vtran_network(sum_h_v)
        targets = self.batch['r'] + self.gamma * (1 - self.batch['dw']) * qtran_opt_next

        l_td = (qtran - targets.detach()) ** 2
        l_opt = (sum_q_evals_opt - qtran_opt.detach() + vtran) ** 2
        l_nopt = torch.clamp(sum_q_evals - qtran.detach() + vtran, max = 0.0) ** 2
        loss = (l_td + self.lambda_opt * l_opt + self.lambda_nopt * l_nopt) * self.batch['active']
        loss = loss.sum() / self.batch['active'].sum()

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


