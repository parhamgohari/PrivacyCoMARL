import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, obs, avail_a, last_onehot_a, a, r, dw, active, episode_len):
        self.obs = obs
        self.avail_a = avail_a
        self.last_onehot_a = last_onehot_a
        self.a = a
        self.r = r
        self.dw = dw
        self.active = active
        self.episode_len = episode_len

    def __len__(self):
        return self.obs.shape[0]
    
    def __getitem__(self, idx):
        return self.obs[idx], self.avail_a[idx], self.last_onehot_a[idx], self.a[idx], self.r[idx], self.dw[idx], self.active[idx], self.episode_len[idx]


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
        self.device = args.device
        self.number_of_appends = 0
        
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
        self.buffer['avail_a'][self.episode_num][episode_step] = avail_a
        
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.number_of_appends += 1
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, seed, method):
        # Randomly sampling
        np.random.seed(seed)
        if method == 'uniform':
            index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        elif method == 'poisson':
            # create an np array with shape (self.current_size, 1) whose elements are a random number between 0 and 1
            # then, for each element, if the value is less than self.batch_size/self.buffer_size, choose its corresponding element in the buffer
            # this is a way to sample from a poisson distribution
            index = np.argwhere(np.random.rand(self.current_size) < self.batch_size/self.buffer_size).flatten()
            
        if index.shape[0] == 0:
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
    
    def clear(self):
        self.episode_num = 0
        self.current_size = 0
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

    def data_loader(self, seed):
        dataset = CustomDataset(
            obs=self.buffer['obs'][:self.current_size],
            # s=self.buffer['s'][:self.current_size],
            avail_a=self.buffer['avail_a'][:self.current_size],
            last_onehot_a=self.buffer['last_onehot_a'][:self.current_size],
            a=self.buffer['a'][:self.current_size],
            r=self.buffer['r'][:self.current_size],
            dw=self.buffer['dw'][:self.current_size],
            active=self.buffer['active'][:self.current_size],
            episode_len=self.episode_len[:self.current_size]
        )
        # fix the seed for the torch data loader
        torch.manual_seed(seed)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
