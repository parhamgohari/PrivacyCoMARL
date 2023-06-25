import torch
from utils.network import QMIX_Net, Q_network, Encoder, orthogonal_init
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from base.agent_base import Agent_Base, BaseGradSampleModule

class CustomGradSampleModule(BaseGradSampleModule):
    def __init__(self, module):
        super().__init__(module)
        self.module = module
    
    def compute_weights_and_biases(self, *args, **kwargs):
        return self.module.compute_weights_and_biases(*args, **kwargs)



class QMIX_Agent(Agent_Base):
    def __init__(self, args, id, seed):
        super(QMIX_Agent, self).__init__(args, id, seed)

        self.hyper_input_dim = args.hypernet_input_dim
        self.q_mix_hidden_dim = args.qmix_hidden_dim

        self.q_network = Q_network(args, self.input_dim)
        self.target_q_network = Q_network(args, self.input_dim)
        self.qmix_network = QMIX_Net(args, self.hyper_input_dim)
        self.target_qmix_network = QMIX_Net(args, self.hyper_input_dim)
        self.state_encoder = Encoder(args, self.input_dim, self.hyper_input_dim)
        self.target_state_encoder = Encoder(args, self.input_dim, self.hyper_input_dim)

        if self.use_anchoring:
            self.anchor_q_network = Q_network(args, self.input_dim)

        if self.use_orthogonal_init:
            orthogonal_init(self.q_network)
            orthogonal_init(self.qmix_network)
            orthogonal_init(self.state_encoder)

        self.eval_parameters = list(self.q_network.parameters()) + list(self.qmix_network.parameters()) + list(self.state_encoder.parameters())

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
            self.qmix_network = CustomGradSampleModule(self.qmix_network)
            self.state_encoder = BaseGradSampleModule(self.state_encoder)
            self.target_q_network = BaseGradSampleModule(self.target_q_network)
            self.target_qmix_network = CustomGradSampleModule(self.target_qmix_network)
            self.target_state_encoder = BaseGradSampleModule(self.target_state_encoder)

            if self.use_anchoring:
                self.anchor_q_network = CustomGradSampleModule(self.anchor_q_network)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_qmix_network.load_state_dict(self.qmix_network.state_dict())
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())

        if self.use_anchoring:
            self.update_anchor_params()

        self.q_network.to(self.device)
        self.qmix_network.to(self.device)
        self.state_encoder.to(self.device)
        self.target_q_network.to(self.device)
        self.target_qmix_network.to(self.device)
        self.target_state_encoder.to(self.device)
        
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

            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.q_network(self.inputs[:, -1].unsqueeze(1)).reshape(self.current_batch_size, 1, -1)
                q_evals_next = torch.cat([self.q_logits[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,action_dim)
                q_evals_next[self.batch['avail_a'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, 1)
                self.q_targets = torch.gather(self.q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len)
            else:
                self.q_targets[self.batch['avail_a'][:, 1:] == 0] = -999999
                self.q_targets = self.q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len)
        
        self.q_evals = torch.gather(self.q_logits, dim=-1, index=self.batch['a'].unsqueeze(-1)).squeeze(-1) 
        
        self.w1, self.b1, self.w2, self.b2 = self.qmix_network.compute_weights_and_biases(self.state_encoder(self.inputs[:, :-1]))
        self.target_w1, self.target_b1, self.target_w2, self.target_b2 = self.target_qmix_network.compute_weights_and_biases(self.target_state_encoder(self.inputs[:, 1:]).detach())    


    def peer2peer_messaging_for_computing_qmix_pt1(self, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.message_to_send = torch.zeros((self.current_batch_size * self.max_episode_len, 1, 2 * self.q_mix_hidden_dim, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size * self.max_episode_len, 1, 2 * self.q_mix_hidden_dim, self.n_agents), device=self.device, dtype=torch.float32)
            self.secret_1 = torch.bmm(self.q_evals.view(-1, 1, 1), self.w1) + self.b1
            self.secret_2 = torch.bmm(self.q_targets.view(-1, 1, 1), self.target_w1) + self.target_b1
            self.secret = torch.cat([self.secret_1, self.secret_2], dim = -1)
            
            if self.use_secret_sharing:
                self.update_seed()
                self.secret_shares = self.encrypt(self.secret.clone().detach())
            else:
                self.secret_shares = self.secret.clone().detach().unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents

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
            sum_first_layer_pre_activation_no_grad = self.decrypt(sender_message) if self.use_secret_sharing else sender_message

            self.intermediate_result_1 = torch.nn.functional.elu(sum_first_layer_pre_activation_no_grad[:, :, :self.q_mix_hidden_dim] - self.secret_1.detach() + self.secret_1)
            self.intermediate_result_2 = torch.nn.functional.elu(sum_first_layer_pre_activation_no_grad[:, :, self.q_mix_hidden_dim:])
        
    def peer2peer_messaging_for_computing_qmix_pt2(self, mode, sender_id = None, sender_message = None):
        if '1' in mode:
            self.message_to_send = torch.zeros((self.current_batch_size * self.max_episode_len, 1, 2, self.n_agents), device=self.device, dtype=torch.float32)
            self.message_to_rece = torch.zeros((self.current_batch_size * self.max_episode_len, 1, 2, self.n_agents), device=self.device, dtype=torch.float32)
            self.secret_1 = torch.bmm(self.intermediate_result_1, self.w2) + self.b2
            self.secret_2 = torch.bmm(self.intermediate_result_2, self.target_w2) + self.target_b2
            self.secret = torch.cat([self.secret_1, self.secret_2], dim = -1)
            
            if self.use_secret_sharing:
                self.update_seed()
                self.secret_shares = self.encrypt(self.secret.clone().detach())
            else:
                self.secret_shares = self.secret.clone().detach().unsqueeze(-1).repeat(1, 1, 1, self.n_agents) / self.n_agents

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
            sum_second_layer_no_grad = self.decrypt(sender_message) if self.use_secret_sharing else sender_message

            q_total = sum_second_layer_no_grad[:, :, 0].unsqueeze(-1) - self.secret_1.detach() + self.secret_1
            self.q_total = q_total.view(self.current_batch_size, self.max_episode_len, 1)
            target_q_total = sum_second_layer_no_grad[:, :, 1].unsqueeze(-1)
            self.target_q_total = target_q_total.view(self.current_batch_size, self.max_episode_len, 1).detach()


    def train(self, total_steps):
        targets = self.batch['r'] + self.gamma * (1-self.batch['dw']) * self.target_q_total
        l_td = self.q_total - targets.detach()
        l_td = l_td * self.batch['active']
        loss = (l_td ** 2).sum() / self.batch['active'].sum() 

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
                self.state_encoder.module.hidden = None
                self.target_q_network.module.encoder.hidden = None
                self.target_state_encoder.module.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.module.encoder.hidden = None
            else:
                self.q_network.encoder.hidden = None
                self.state_encoder.hidden = None
                self.target_q_network.encoder.hidden = None
                self.target_state_encoder.hidden = None
                if self.use_anchoring:
                    self.anchor_q_network.encoder.hidden = None
        self.last_onehot_a = torch.zeros((self.action_dim), device=self.device)

    def update_target_networks(self):
        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                self.target_qmix_network.load_state_dict(self.qmix_network.state_dict())
                self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
                if self.use_anchoring:
                    self.anchor_q_network.load_state_dict(self.q_network.state_dict())
        else:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.qmix_network.parameters(), self.target_qmix_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.state_encoder.parameters(), self.target_state_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
