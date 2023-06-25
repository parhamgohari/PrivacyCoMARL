import torch
from utils.buffer import ReplayBuffer
from opacus.privacy_engine import GradSampleModule

class BaseGradSampleModule(GradSampleModule):
    def __init__(self, module):
        super().__init__(module)
        self.module = module

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
        self.use_orthogonal_init = args.use_orthogonal_init
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

        self.input_dim = self.obs_dim
        if self.add_last_action:
            self.input_dim += self.action_dim

        self.Q = args.Q
        self.precision = args.precision
        self.base = args.base

    def choose_action(self, epsilon, evaluate_anchor_q = False):
        local_obs = self.observation
        avail_a = self.avail_action
        last_onehot_a = self.last_onehot_a
        with torch.no_grad():
            self.update_seed()
            if torch.rand(1).item() < epsilon:
                a = torch.where(torch.tensor(avail_a)==True)[0][torch.randint(0, sum(avail_a), (1,)).item()].item()
            else:
                inputs = []
                obs = torch.tensor(local_obs, dtype=torch.float32, device=self.device)
                inputs.append(obs)
                if self.add_last_action:
                    inputs.append(last_onehot_a)
                inputs = torch.cat(inputs, dim=0)

                if evaluate_anchor_q:
                    assert self.use_anchoring
                    q_values = self.anchor_q_network(inputs.view(1, 1, -1)).view(-1)
                else:
                    q_values = self.q_network(inputs.view(1, 1, -1)).view(-1)
                avail_a = torch.tensor(avail_a, dtype=torch.float32, device=self.device)
                q_values[avail_a == 0.0] = -float('inf')
                a = torch.max(q_values, dim=0)[1].item()
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
        shares = torch.stack(shares, dim=-1)
        return shares
    
    def decrypt(self, shares):
        return self.decoder(torch.sum(shares, dim=0) % self.Q)
    
    def update_target_networks(self):
        raise NotImplementedError

    def update_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            self.seed = seed
        else:
            torch.manual_seed(self.seed)
            self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    def update_anchor_params(self):
        self.anchor_q_network.load_state_dict(self.q_network.state_dict())

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
        raise NotImplementedError

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

    def train(self):
        raise NotImplementedError