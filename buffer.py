import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.id = args.id
        self.obs_dim = args.obs_dim
        # self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0
        self.use_rnn = args.use_rnn
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.device = args.device
        self.use_poisson_sampling = args.use_poisson_sampling
        
        self.buffer = {'obs': np.zeros([self.buffer_size, self.episode_limit + 1, self.obs_dim]),
                #    's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                    'avail_a': np.ones([self.buffer_size, self.episode_limit + 1, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                    'last_onehot_a': np.zeros([self.buffer_size, self.episode_limit + 1, self.action_dim]),
                    'a': np.zeros([self.buffer_size, self.episode_limit]),
                    'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                    'dw': np.ones([self.buffer_size, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'dw'
                    'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                    }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs, avail_a, last_onehot_a, a, r, dw, rnn_hidden=None):
        self.buffer['obs'][self.episode_num][episode_step] = obs
        # self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a'][self.episode_num][episode_step] = avail_a
        self.buffer['last_onehot_a'][self.episode_num][episode_step + 1] = last_onehot_a
        self.buffer['a'][self.episode_num][episode_step] = a
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw
        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_step(self, episode_step, obs, avail_a, rnn_hidden=None):
        self.buffer['obs'][self.episode_num][episode_step] = obs
        # self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a'][self.episode_num][episode_step] = np.ones(self.action_dim)

        self.buffer['obs'][self.episode_num][episode_step + 1:] = np.zeros((self.episode_limit - episode_step, self.obs_dim))
        # self.buffer['s'][self.episode_num][episode_step + 1:] = np.zeros((self.episode_limit + 1 - episode_step, self.state_dim))
        self.buffer['avail_a'][self.episode_num][episode_step + 1:] = np.ones((self.episode_limit - episode_step, self.action_dim))
        self.buffer['last_onehot_a'][self.episode_num][episode_step + 1:] = np.zeros((self.episode_limit - episode_step, self.action_dim))
        self.buffer['a'][self.episode_num][episode_step:] = np.zeros((self.episode_limit - episode_step,))
        self.buffer['r'][self.episode_num][episode_step:] = np.zeros((self.episode_limit - episode_step, 1))
        self.buffer['dw'][self.episode_num][episode_step:] = np.ones((self.episode_limit - episode_step, 1))
        self.buffer['active'][self.episode_num][episode_step:] = np.zeros((self.episode_limit - episode_step, 1))

        
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, seed):
        # Randomly sampling
        torch.manual_seed(seed)
        if not self.use_poisson_sampling:
            index = torch.randint(0, self.current_size, size=(self.batch_size, ))
        else:
            index = torch.where(torch.rand(self.current_size) < self.batch_size/self.buffer_size)[0]

        if index.shape[0] ==0:
            return None, None, None
        
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs' or key == 'avail_a' or key == 'last_onehot_a':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32, device=self.device)
            elif key == 'a':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long, device=self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32, device=self.device)

        return batch, max_episode_len, index.shape[0]