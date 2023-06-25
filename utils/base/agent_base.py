import torch
from utils.buffer import ReplayBuffer
from opacus.accountants import RDPAccountant
from utils.network import Q_network, QMIX_Net, Encoder, orthogonal_init
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.privacy_engine import GradSampleModule
from utils.secret_sharing import Secret_Sharing_Module

class Agent_Base(object):
    def __init__(self, args, id, seed):
        args.id = id
        self.id = id
        self.n_agents = args.n_agents
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.add_last_action = args.add_last_action
        self.use_rnn = args.use_rnn
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
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_Adam = args.use_Adam
        self.use_SGD = args.use_SGD
        self.use_lr_decay = args.use_lr_decay
        self.use_secret_sharing = args.use_secret_sharing
        self.use_poisson_sampling = args.use_poisson_sampling
        self.use_dp = args.use_dp
        self.noise_multiplier = args.noise_multiplier
        self.device = args.device
        self.buffer_throughput = args.buffer_throughput
        self.use_anchoring = args.use_anchoring
        self.train_step = 0
        self.update_seed(seed)

        self.replay_buffer = ReplayBuffer(args)

        if self.use_secret_sharing:
            self.secret_sharing_module = Secret_Sharing_Module(self.n_agents, self.device)

        self.input_dim = self.obs_dim
        if self.add_last_action:
            self.input_dim += self.action_dim

        self.running_models = {'q_network': Q_network(args, self.input_dim)}
        self.target_models = {'q_network': Q_network(args, self.input_dim)}
        
        if self.use_anchoring:
            self.anchor_models = {'q_network': Q_network(args, self.input_dim)}

        for key in self.running_models.keys():
            self.running_models[key].to(self.device)
            self.target_models[key].to(self.device)
            if self.use_anchoring:
                self.anchor_models[key].to(self.device)
            self.target_models[key].load_state_dict(self.running_models[key].state_dict())
            if self.use_anchoring:
                self.anchor_models[key].load_state_dict(self.running_models[key].state_dict())


    def init_optimizer(self, weight_decay = 0.0, momentum = 0.0):
        self.eval_params = []
        for _, model in self.running_models.items():
            self.eval_params += list(model.parameters())
        if self.use_RMS:
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr = self.lr)
        elif self.use_Adam:
            self.optimizer = torch.optim.Adam(self.eval_params, lr = self.lr)
        elif self.use_SGD:
            self.optimizer = torch.optim.SGD(self.eval_params, lr = self.lr, momentum=momentum, weight_decay=weight_decay)
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
            
            for model_names in self.running_models.keys():
                self.running_models[model_names] = GradSampleModule(self.running_models[model_names])
                self.running_models[model_names].to(self.device)
                self.target_models[model_names] = GradSampleModule(self.target_models[model_names])
                self.target_models[model_names].to(self.device)
                if self.use_anchoring:
                    self.anchor_models[model_names] = GradSampleModule(self.anchor_models[model_names])
                    self.anchor_models[model_names].to(self.device)


    def choose_action(self, *, local_obs, last_onehot_a, avail_a, epsilon, evaluate_anchor_q = False):
        with torch.no_grad():
            self.update_seed()
            if torch.rand(1).item() < epsilon:
                a = torch.where(torch.tensor(avail_a)==True)[0][torch.randint(0, sum(avail_a), (1,)).item()].item()
            else:
                inputs = []
                obs = torch.tensor(local_obs, dtype=torch.float32, device=self.device)
                inputs.append(obs)
                if self.add_last_action:
                    last_a = torch.tensor(last_onehot_a, dtype=torch.float32, device=self.device)
                    inputs.append(last_a)
                inputs = torch.cat(inputs, dim=0)

                if evaluate_anchor_q:
                    assert self.use_anchoring
                    q_values = self.anchor_models['q_network'](inputs.unsqueeze(0)).squeeze(0)
                else:
                    q_values = self.running_models['q_network'](inputs.unsqueeze(0)).squeeze(0)
                avail_a = torch.tensor(avail_a, dtype=torch.float32, device=self.device)
                q_values[avail_a == 0.0] = -float('inf')
                a = torch.max(q_values, dim=0)[1].item()
            self.current_a = a
        return a
    
    def train(self):
        raise NotImplementedError
    
    def update_target_networks(self):
        if self.use_hard_update:
            for key in self.running_models.keys():
                self.target_models[key].load_state_dict(self.running_models[key].state_dict())
        else:
            for key in self.running_models.keys():
                for target_param, param in zip(self.target_models[key].parameters(), self.running_models[key].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            self.seed = seed
        else:
            torch.manual_seed(self.seed)
            self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    def update_anchor_params(self):
        for key in self.running_models.keys():
            self.anchor_models[key].load_state_dict(self.running_models[key].state_dict())

    def lr_decay(self, total_steps):
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
    
    def reset(self):
        if self.use_rnn:
            self.q_network.encoder.hidden = None
            self.target_q_network.encoder.hidden = None
            if self.use_anchoring:
                self.anchor_q_network.encoder.hidden = None
        
        self.last_onehot_a = torch.zeros((self.action_dim), device=self.device)

    def receive_obs(self, obs):
        self.observation = obs

    def receive_avail_a(self, avail_a):
        self.avail_action = avail_a

    def store_transition(self, episode_step, dw, r):
        self.replay_buffer.store_transition(
            episode_step = episode_step,
            obs = self.observation,
            avail_a = self.avail_action,
            last_onehot_a = self.last_onehot_a,
            a = self.current_a,
            r = r,
            dw = dw
        )
        self.last_onehot_a = torch.eye(self.action_dim)[self.current_a].to(self.device)

    def store_last_step(self, episode_step):
        self.replay_buffer.store_last_step(
            episode_step = episode_step,
            obs = self.observation,
            avail_a = self.avail_action
        )