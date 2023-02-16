import torch
from smac.env import StarCraft2Env
import numpy as np
import argparse
from agent import VDN_Agent
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



class VDN_Runner(object):
    def __init__(self, args, env_name, exp_id, seed, privacy_mechanism = None):
        self.args = args
        self.env_name = env_name
        self.exp_id = exp_id
        self.device = args.device
        self.seed = seed
        self.privacy_mechanism = privacy_mechanism
        self.env = StarCraft2Env(map_name=env_name, seed = self.seed)
        self.env_info = self.env.get_env_info()
        self.args.n_agents = self.env_info["n_agents"]
        self.n_agents = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.n_agents))
        print("obs_dim={}".format(self.args.obs_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.epsilon = self.args.epsilon

        _, self.seed = self.branch_seed(self.seed)
        # create n_agents VDN agents
        self.agents = [VDN_Agent(self.args, id, self.seed) for id in range(self.args.n_agents)]

        # creat a tensorboard
        self.writer = SummaryWriter(log_dir="./log/{}_{}_{}".format(self.args.algorithm, env_name, exp_id))

        self.win_rates = []
        self.total_steps = 0

    def branch_seed(self, current_seed, number_of_branches = 1):
        """
        This function is used to generate different seeds for different branches
        """
        np.random.seed(current_seed)
        random_seeds = np.random.randint(0, 2 ** 32 - 1, number_of_branches + 1)
        return random_seeds[:-1], random_seeds[-1]

    def run(self):
        evaluate_num = -1
        pbar = tqdm(total = self.total_steps)
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate = False)
            self.total_steps += episode_steps
            pbar.update(episode_steps)

            
            # start training if all agent replay buffer has enough samples
            if min([agent.replay_buffer.current_size for agent in self.agents]) >= self.args.batch_size:
                #create an np array of size batch_size * n_agents (sender) * n_agents (receiver)
                
                for agent in self.agents:
                    agent.peer2peer_messaging(self.seed, mode = '0. initiate')
                for agent in self.agents:
                    agent.peer2peer_messaging(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:, receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)
                sum_shares = torch.sum(local_sums, dim = 0)
                        
                for agent in self.agents:
                    agent.train(self.total_steps, sum_shares)
                _, self.seed = self.branch_seed(self.seed)


        self.evaluate_policy()
        self.env.close()
        tqdm.close()

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        # Save the win rates
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.exp_id, self.seed), np.array(self.win_rates))

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        for agent in self.agents:
            if self.args.use_rnn:
                agent.q_network.init_hidden()
        joint_last_onehot_a = np.zeros((self.n_agents, self.args.action_dim))

        for episode_step in range(self.args.episode_limit):
            # get observations
            joint_obs = self.env.get_obs()
            joint_avail_a = self.env.get_avail_actions()
            epsilon = 0 if evaluate else self.epsilon
            joint_action = [agent.choose_action(joint_obs[agent.id], joint_last_onehot_a[agent.id], joint_avail_a[agent.id], epsilon) for agent in self.agents]
            joint_last_onehot_a = np.eye(self.args.action_dim)[joint_action]

            r, done, info = self.env.step(joint_action)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:    
                    dw = False

                for agent in self.agents:
                    
                    agent.replay_buffer.store_transition(
                        episode_step = episode_step,
                        obs = joint_obs[agent.id],
                        avail_a = joint_avail_a[agent.id],
                        last_onehot_a = joint_last_onehot_a[agent.id],
                        a = joint_action[agent.id],
                        r = r,
                        dw = dw
                        )
                        

                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            joint_obs = self.env.get_obs()
            joint_avail_a = self.env.get_avail_actions()
            for agents in self.agents:
                agents.replay_buffer.store_last_step(
                    episode_step = episode_step,
                    obs = joint_obs[agents.id],
                    avail_a = joint_avail_a[agents.id]
                    )

        return win_tag, episode_reward, episode_step + 1
    


            

        
                







if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="VDN", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    # check if cuda available
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("mps")

    env_names = ['3m', '8m', '2s3z']
    env_index = 0
    runner = VDN_Runner(args, env_name=env_names[env_index], exp_id=1, seed=0)
    runner.run()
        
