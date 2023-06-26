import torch
from utils.network import Q_network, Encoder, orthogonal_init
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from base.agent_base import Agent_Base, BaseGradSampleModule

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

        self.state_encoder = Encoder(args, self.input_dim, self.vtran_hidden_dim)
        self.target_state_encoder = Encoder(args, self.input_dim, self.vtran_hidden_dim)
        self.state_action_encoder = Encoder(args, self.input_dim + self.action_dim, self.qtran_hidden_dim)
        self.target_state_action_encoder = Encoder(args, self.input_dim + self.action_dim, self.qtran_hidden_dim)


        self.qtran_network_layer_1 = torch.nn.Linear(self.qtran_hidden_dim, self.qtran_hidden_dim)
        self.qtran_network_layer_2 = torch.nn.Linear(self.qtran_hidden_dim, 1)
        self.target_qtran_network_layer_1 = torch.nn.Linear(self.qtran_hidden_dim, self.qtran_hidden_dim)
        self.target_qtran_network_layer_2 = torch.nn.Linear(self.qtran_hidden_dim, 1)

        self.vtran_network_layer_1 = torch.nn.Linear(self.vtran_hidden_dim, self.vtran_hidden_dim)
        self.vtran_network_layer_2 = torch.nn.Linear(self.vtran_hidden_dim, 1)
        self.target_vtran_network_layer_1 = torch.nn.Linear(self.vtran_hidden_dim, self.vtran_hidden_dim)
        self.target_vtran_network_layer_2 = torch.nn.Linear(self.vtran_hidden_dim, 1)

        if self.use_anchoring:
            self.anchor_q_network = Q_network(args, self.input_dim)

        if self.use_orthogonal_init:
            orthogonal_init(self.q_network)
            orthogonal_init(self.state_encoder)
            orthogonal_init(self.state_action_encoder)
            orthogonal_init(self.qtran_network_layer_1)
            orthogonal_init(self.qtran_network_layer_2)
            orthogonal_init(self.vtran_network_layer_1)
            orthogonal_init(self.vtran_network_layer_2)

        self.eval_parameters = list(self.q_network.parameters()) + list(self.qtran_network_layer_1.parameters()) + list(self.qtran_network_layer_2.parameters()) + list(self.vtran_network_layer_1.parameters()) + list(self.vtran_network_layer_2.parameters()) + list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters())

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
            self.state_encoder = BaseGradSampleModule(self.state_encoder)
            self.target_state_encoder = BaseGradSampleModule(self.target_state_encoder)
            self.state_action_encoder = BaseGradSampleModule(self.state_action_encoder)
            self.target_state_action_encoder = BaseGradSampleModule(self.target_state_action_encoder)
            self.qtran_network_layer_1 = BaseGradSampleModule(self.qtran_network_layer_1)
            self.qtran_network_layer_2 = BaseGradSampleModule(self.qtran_network_layer_2)
            self.target_qtran_network_layer_1 = BaseGradSampleModule(self.target_qtran_network_layer_1)
            self.target_qtran_network_layer_2 = BaseGradSampleModule(self.target_qtran_network_layer_2)
            self.vtran_network_layer_1 = BaseGradSampleModule(self.vtran_network_layer_1)
            self.vtran_network_layer_2 = BaseGradSampleModule(self.vtran_network_layer_2)
            self.target_vtran_network_layer_1 = BaseGradSampleModule(self.target_vtran_network_layer_1)
            self.target_vtran_network_layer_2 = BaseGradSampleModule(self.target_vtran_network_layer_2)

            if self.use_anchoring:
                self.anchor_q_network = BaseGradSampleModule(self.anchor_q_network)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_state_action_encoder.load_state_dict(self.state_action_encoder.state_dict())
        self.target_qtran_network_layer_1.load_state_dict(self.qtran_network_layer_1.state_dict())
        self.target_qtran_network_layer_2.load_state_dict(self.qtran_network_layer_2.state_dict())
        self.target_vtran_network_layer_1.load_state_dict(self.vtran_network_layer_1.state_dict())
        self.target_vtran_network_layer_2.load_state_dict(self.vtran_network_layer_2.state_dict())

        if self.use_anchoring:
            self.update_anchor_params()

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.state_encoder.to(self.device)
        self.target_state_encoder.to(self.device)
        self.state_action_encoder.to(self.device)
        self.target_state_action_encoder.to(self.device)
        self.qtran_network_layer_1.to(self.device)
        self.qtran_network_layer_2.to(self.device)
        self.target_qtran_network_layer_1.to(self.device)
        self.target_qtran_network_layer_2.to(self.device)
        self.vtran_network_layer_1.to(self.device)
        self.vtran_network_layer_2.to(self.device)
        self.target_vtran_network_layer_1.to(self.device)
        self.target_vtran_network_layer_2.to(self.device)
        
        if self.use_anchoring:
            self.anchor_q_network.to(self.device)

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
        

        self.q_logits = self.q_network(self.inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,action_dim)
        
        with torch.no_grad():
            
            self.q_target_logits = self.target_q_network(self.inputs)
            self.q_targets = self.q_target_logits[:, 1:]
            self.a_opt = torch.argmax(self.q_logits, dim = -1)

            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.q_network(self.inputs[:, -1].unsqueeze(1)).reshape(self.current_batch_size, 1, -1)
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

        self.h_q = self.state_action_encoder(torch.cat([self.inputs[:, :-1], self.onehot_a], dim=-1))
        self.h_q_opt = self.state_action_encoder(torch.cat([self.inputs[:, :-1], self.onehot_a_opt], dim=-1))
        self.h_q_opt_next = self.target_state_action_encoder(torch.cat([self.inputs[:, 1:], self.onehot_a_opt_next], dim=-1))
        self.h_v = self.state_encoder(self.inputs[:, :-1])

    def peer2peer_messaging_pt1(self, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.secret = []
            qtran_first_layer_preactivation = self.qtran_network_layer_1(self.h_q)
            self.secret.append(qtran_first_layer_preactivation)
            self.h_q_start = 0
            self.h_q_end = self.qtran_hidden_dim
            
            qtran_opt_first_layer_preactivation = self.qtran_network_layer_1(self.h_q_opt)
            self.secret.append(qtran_opt_first_layer_preactivation)
            self.h_q_opt_start = self.h_q_end
            self.h_q_opt_end = self.h_q_opt_start + self.qtran_hidden_dim

            qtran_opt_next_first_layer_preactivation = self.target_qtran_network_layer_1(self.h_q_opt_next)
            self.secret.append(qtran_opt_next_first_layer_preactivation)
            self.h_q_opt_next_start = self.h_q_opt_end
            self.h_q_opt_next_end = self.h_q_opt_next_start + self.qtran_hidden_dim

            vtran_first_layer_preactivation = self.vtran_network_layer_1(self.h_v)
            self.secret.append(vtran_first_layer_preactivation)
            self.h_v_start = self.h_q_opt_next_end
            self.h_v_end = self.h_v_start + self.vtran_hidden_dim

            self.secret.append(self.q_evals.unsqueeze(-1))
            self.q_evals_start = self.h_v_end
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
            self.qtran_first_layer = torch.nn.functional.relu(self.sum_secrets[:, :, self.h_q_start:self.h_q_end])
            self.qtran_opt_first_layer = torch.nn.functional.relu(self.sum_secrets[:, :, self.h_q_opt_start:self.h_q_opt_end])
            self.qtran_opt_next_first_layer = torch.nn.functional.relu(self.sum_secrets[:, :, self.h_q_opt_next_start:self.h_q_opt_next_end])
            self.vtran_first_layer = torch.nn.functional.relu(self.sum_secrets[:, :, self.h_v_start:self.h_v_end])
            self.sum_q_evals = self.sum_secrets[:, :, self.q_evals_start:self.q_evals_end]
            self.sum_q_evals_opt = self.sum_secrets[:, :, self.q_evals_opt_start:self.q_evals_opt_end]

    def peer2peer_messaging_pt2(self, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.secret = []

            qtran_second_layer_preactivation = self.qtran_network_layer_2(self.qtran_first_layer)
            self.secret.append(qtran_second_layer_preactivation)

            qtran_opt_second_layer_preactivation = self.qtran_network_layer_2(self.qtran_opt_first_layer)
            self.secret.append(qtran_opt_second_layer_preactivation)

            qtran_opt_next_second_layer_preactivation = self.target_qtran_network_layer_2(self.qtran_opt_next_first_layer)
            self.secret.append(qtran_opt_next_second_layer_preactivation)

            vtran_second_layer_preactivation = self.vtran_network_layer_2(self.vtran_first_layer)
            self.secret.append(vtran_second_layer_preactivation)

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
            self.qtran = self.sum_secrets[:, :, 0].unsqueeze(-1)
            self.qtran_opt = self.sum_secrets[:, :, 1].unsqueeze(-1)
            self.qtran_opt_next = self.sum_secrets[:, :, 2].unsqueeze(-1)
            self.vtran = self.sum_secrets[:, :, 3].unsqueeze(-1)

    def train(self, total_steps):
        targets = self.batch['r'] + self.gamma * (1 - self.batch['dw']) * self.qtran_opt_next

        l_td = (self.qtran - targets.detach()) ** 2 
        l_opt = (self.sum_q_evals_opt - self.qtran_opt.detach() + self.vtran) ** 2
        l_nopt = torch.clamp(self.sum_q_evals - self.qtran.detach() + self.vtran, max = 0.0) ** 2
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
    
    def reset(self):
        if self.use_rnn:
            if self.use_dp:
                self.q_network.module.encoder.hidden = None
                self.target_q_network.module.encoder.hidden = None
                self.state_encoder.module.hidden = None
                self.target_state_encoder.module.hidden = None
                self.state_action_encoder.module.hidden = None
                self.target_state_action_encoder.module.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.module.encoder.hidden = None
            else:
                self.q_network.encoder.hidden = None
                self.target_q_network.encoder.hidden = None
                self.state_encoder.hidden = None
                self.target_state_encoder.hidden = None
                self.state_action_encoder.hidden = None
                self.target_state_action_encoder.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.encoder.hidden = None
            
        self.last_onehot_a = torch.zeros((self.action_dim), device=self.device)

    def update_target_networks(self):
        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
                self.target_state_action_encoder.load_state_dict(self.state_action_encoder.state_dict())
                self.target_qtran_network_layer_1.load_state_dict(self.qtran_network_layer_1.state_dict())
                self.target_qtran_network_layer_2.load_state_dict(self.qtran_network_layer_2.state_dict())
                self.target_vtran_network_layer_1.load_state_dict(self.vtran_network_layer_1.state_dict())
                self.target_vtran_network_layer_2.load_state_dict(self.vtran_network_layer_2.state_dict())
        else:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.state_encoder.parameters(), self.target_state_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.state_action_encoder.parameters(), self.target_state_action_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qtran_network_layer_1.parameters(), self.target_qtran_network_layer_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qtran_network_layer_2.parameters(), self.target_qtran_network_layer_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.vtran_network_layer_1.parameters(), self.target_vtran_network_layer_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.vtran_network_layer_2.parameters(), self.target_vtran_network_layer_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                




