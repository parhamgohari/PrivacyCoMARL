import collections
import numpy as np


"""a replay buffer with fixed size whose elements are like the following:
(tranition id, (last_action, last_local_obs), team_reward, terminated, action, (action, local_obs))
"""
class ReplayBuffer:
    def __init__(self, buffer_size, minibatch_size, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = collections.deque(maxlen=buffer_size)
        self.transition_id = 0
        self.episode_id = 0
    
    def __len__(self):
        return len(self.buffer)

    def store(self, transition):
        self.buffer.append(transition)
        print("tranition {} is stored in buffer".format(transition[0]))
        print(transition)
    
    def sample(self, batch_size, element_index_list = None):
        if element_index_list is None:
            element_index_list = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in element_index_list]

    

