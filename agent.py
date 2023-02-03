from buffer import ReplayBuffer
from network import Q_network_MLP, Q_network_RNN
import torch
import numpy as np

class AgentBase(object):
    def __init__(self, args) -> None:
        self.N = 1
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  # 这里的batch_size代表有多少个episode
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.use_history_encoding = args.use_history_encoding

        self.current_action = 0
        self.last_action = 0
        self.last_obs = np.zeros(self.obs_dim)
        self.last_q_value = 0

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, self.obs_dim, self.action_dim)


        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
            self.last_onehot_a = np.zeros(self.action_dim)
        
        if self.use_rnn:
            print("------use RNN------")
            self.eval_Q_net = Q_network_RNN(args, self.input_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim)
        else:
            print("------use MLP------")
            self.eval_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        self.eval_parameters = list(self.eval_Q_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)
        
        self.train_step = 0

    def receive_observation(self, obs):
        self.current_obs = obs
        self.last_action = self.current_action

    def receive_next_observation(self, obs):
        self.last_obs = self.current_obs
        self.current_obs = obs

    def receive_reward(self, reward):
        self.current_reward = reward
    
    def receive_terminated(self, terminated):
        self.current_terminated = terminated
    
    """get q values of all actions"""
    def get_q_values(self, obs, avail_a, last_onehot_a = None):
        with torch.no_grad():
            inputs = []
            obs = torch.tensor(obs, dtype = torch.float32)
            inputs.append(obs)
            if self.add_last_action:
                if last_onehot_a is None:
                    last_onehot_a = torch.functional.one_hot(torch.tensor(self.last_action), self.action_dim)
                inputs.append(last_onehot_a)

            inputs = torch.cat([x for x in inputs], dim = -1) #originally the input.shape = (N,inputs_dim)
            q_value = self.eval_Q_net(inputs)

            avail_a = torch.tensor(avail_a, dtype = torch.float32)
            q_value[avail_a == 0] = -float('inf')
        
        return q_value.numpy()


    """choose action according to epsilon-greedy policy"""
    def choose_action(self, obs, avail_a, epsilon, last_onehot_a = None):    

        if np.random.uniform() < epsilon:
            a = np.random.choice(np.nonzero(avail_a)[0])
            q_value = self.get_q_values(obs, avail_a, last_onehot_a)[a]
            self.current_action = a
            self.last_q_value = q_value
            return a, q_value
        else:
            self.last_q_value = self.get_q_values(obs, avail_a, last_onehot_a)
            a, q_value = self.last_q_value.argmax(axis = -1), self.last_q_value.max(axis = -1)
            self.last_q_value = q_value
            self.current_action = a   
            return a, q_value

    """store transition in replay buffer"""
    def update_buffer(self, transition_id):
        transition = [transition_id, (self.last_obs, self.last_action), self.current_reward, self.current_terminated, (self.current_obs, self.current_action)]
        self.replay_buffer.store(transition)
    
    """Update the target network"""
    def update_target_net(self):
        if self.use_hard_update:
            self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
        else:
            for eval_param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)




    


class AgentVDN(AgentBase):
    def __init__(self) -> None:
        super(AgentVDN, self).__init__()

    def send_message(self, minibatch, method):
        raise NotImplementedError

    def receive_message(self, method):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

class AgentQMIX(AgentBase):
    def __init__(self) -> None:
        super(AgentQMIX, self).__init__()

    def send_message(self, minibatch, method):
        raise NotImplementedError

    def receive_message(self, method):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError



    