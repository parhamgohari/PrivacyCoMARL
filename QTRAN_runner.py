import torch
from smac.env import StarCraft2Env
import numpy as np
import argparse
from QTRAN_agent import QTRAN_Base_Agent, QTRAN_Alt_Agent
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import json
import datetime



class QTRAN_Runner(object):
    def __init__(self, args, env_name, exp_id, seed, privacy_mechanism = None):
        self.args = args
        self.env_name = env_name
        self.exp_id = exp_id
        self.device = args.device
        self.seed = seed
        self.privacy_mechanism = privacy_mechanism
        self.env = StarCraft2Env(map_name=env_name, seed = self.seed)
        self.update_seed()
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

        if args.use_dp:
            self.privacy_budget = {
                'epsilon': 0,
                'delta': 0,
            }

        if args.use_anchoring:
            raise NotImplementedError
            self.anchor_threshold = 0.5

        # create agents
        if args.algorithm == 'QTRAN-base':
            self.agents = [QTRAN_Base_Agent(self.args, id, self.seed) for id in range(self.args.n_agents)]
        elif args.algorithm == 'QTRAN-alt':
            self.agents = [QTRAN_Alt_Agent(self.args, id, self.seed) for id in range(self.args.n_agents)]
        else:
            raise NotImplementedError
        # creat a tensorboard
        self.writer = SummaryWriter(log_dir="./log/{}_{}_{}".format(self.args.algorithm, env_name, exp_id))

        self.win_rates = []
        self.total_steps = 0

    def update_seed(self):
        """
        This function is used to generate different seeds for different branches
        """
        torch.manual_seed(self.seed)
        self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    def run(self):
        evaluate_num = -1
        pbar = tqdm(total = self.total_steps)
        dp_measured = False
        episode_num = 0
        losses = []
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                if len(losses) != 0:
                    print('-----------------------------------------------------------------------------------')
                    print(losses[-3])
                    print(losses[-2])
                    print(losses[-1])
                    print('-----------------------------------------------------------------------------------')
                    print('\n')
                if self.args.use_anchoring:
                    raise NotImplementedError
                    self.evaluate_policy(use_anchor_q = True)
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate = False)
            episode_num += 1
            self.total_steps += episode_steps
            pbar.update(episode_steps)

            if self.args.use_anchoring:
                raise NotImplementedError
                if self.win_rates[-1] > self.anchor_threshold:
                    for agent in self.agents:
                        agent.anchor_q_network.load_state_dict(agent.q_network.state_dict())
                        agent.anchoring_weight += 1e-2
                    self.anchor_threshold *= 1.3

            
            # start training if all agent replay buffer has enough samples
            if min([agent.replay_buffer.current_size for agent in self.agents]) >= self.args.batch_size:
                #create an np array of size batch_size * n_agents (sender) * n_agents (receiver)
                skip_iteration = False
                for agent in self.agents:
                    agent.initiate_peer2peer_messaging(self.seed)
                    if agent.empty_batch:
                        skip_iteration = True
                        break
                self.update_seed()
                if skip_iteration:
                    continue

                # Compute q_jt_sum
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_q_jt_sum(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_q_jt_sum(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:, receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_q_jt_sum(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_q_jt_sum(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_q_jt_sum(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))

                # Compute q_jt_sum_opt
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_q_jt_sum_opt(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_q_jt_sum_opt(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:, receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_q_jt_sum_opt(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_q_jt_sum_opt(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_q_jt_sum_opt(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))


                # Compute q_jt
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_q_jt(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_q_jt(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_q_jt(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_q_jt(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_q_jt(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))


                # Compute y_dqn
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_y_dqn(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_y_dqn(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_y_dqn(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_y_dqn(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_y_dqn(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))


                # Compute q_jt_opt
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_q_jt_opt(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_q_jt_opt(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_q_jt_opt(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_q_jt_opt(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_q_jt_opt(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))

                # Compute v_jt
                for agent in self.agents:
                    self.update_seed()
                    agent.peer2peer_messaging_for_computing_v_jt(self.seed, mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_v_jt(self.seed, mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_v_jt(self.seed, mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_v_jt(self.seed, mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_v_jt(self.seed, mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))

                # train
                federated_model_grads = [None, None, None]
                for agent in self.agents:
                    verbose = agent.train(self.total_steps)
                    losses.append(verbose)

                

            if self.args.use_dp:
                if min([agent.replay_buffer.current_size for agent in self.agents]) == self.args.buffer_size and not dp_measured:
                    self.privacy_budget = {
                        'epsilon': max([agent.accountant.get_epsilon(self.args.delta) for agent in self.agents]) /self.args.buffer_throughput,
                        'delta': self.args.delta /self.args.buffer_throughput
                    }
                    dp_measured = True

        self.evaluate_policy()
        self.env.close()
        pbar.close()

    def evaluate_policy(self, use_anchor_q = False):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True, use_anchor_q = use_anchor_q)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        if not use_anchor_q:
            self.win_rates.append(win_rate)
            print("total_steps:{} \t win_rate:{:.2f} \t evaluate_reward:{:.2f}".format(self.total_steps, win_rate, evaluate_reward))
            self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        else:
            print("total_steps:{} \t anchor_win_rate:{:.2f} \t anchor_evaluate_reward:{:.2f}".format(self.total_steps, win_rate, evaluate_reward))
            self.writer.add_scalar('anchor_win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        if self.args.use_dp:
            print("privacy budget: epsilon={:.2f} and delta={:.5f}".format(self.privacy_budget['epsilon'], self.privacy_budget['delta']))
        # Save the win rates
        # np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.exp_id, self.seed), np.array(self.win_rates))

    def run_episode_smac(self, evaluate=False, use_anchor_q=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        for agent in self.agents:
            if self.args.use_rnn:
                agent.q_network.rnn_hidden = None

        for episode_step in range(self.args.episode_limit):
            # get observations
            epsilon = 0 if evaluate else self.epsilon
            joint_action = []
            for agent in self.agents:
                self.update_seed()
                joint_action.append(agent.choose_action(self.env.get_obs_agent(agent.id), self.env.get_avail_agent_actions(agent.id), epsilon, self.seed))
            try:
                r, done, info = self.env.step(joint_action)
            except:
                # pause for 30 seconds
                time.sleep(30)
                del self.env
                from smac.env import StarCraft2Env
                self.env = self.env = StarCraft2Env(map_name=self.env_name, seed = self.seed)
                self.env.reset()
                continue
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r


            if not evaluate:
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True

                else:    
                    dw = False

                for agent in self.agents:
                    agent.store_transition(episode_step = episode_step, r = r, dw = dw)
                        
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            for agents in self.agents:
                agents.replay_buffer.store_last_step(
                    episode_step = episode_step+1,
                    obs = self.env.get_obs_agent(agents.id),
                    avail_a = self.env.get_avail_agent_actions(agents.id)
                )


        return win_tag, episode_reward, episode_step + 1
                

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QTRAN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(6e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=20, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--algorithm", type=str, default="QTRAN-alt", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda_opt", type=float, default=0.1, help="The coefficient of the factorizing error for optimal actions regularization term")
    parser.add_argument("--lambda_nopt", type=float, default=1, help="The coefficient of the factorizing error for non-optimal actions regularization term")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="The norm of the gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS")
    parser.add_argument("--use_Adam", type=bool, default=True, help="Whether to use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")
    parser.add_argument("--use_secret_sharing", type=bool, default=True, help="Whether to use secret sharing")
    parser.add_argument("--use_poisson_sampling", type=bool, default=False, help="Whether to use poisson sampling")
    parser.add_argument("--use_dp", type=bool, default=False, help="Whether to use differential privacy")
    parser.add_argument("--noise_multiplier", type=float, default=0.8, help="Noise multiplier")
    parser.add_argument("--delta", type=float, default=None, help="Delta")
    parser.add_argument("--buffer_throughput", type=float, default=1.0, help="Buffer throughput")
    parser.add_argument("--use_anchoring", type=bool, default=False, help="Whether to use anchoring")


    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps



    args.device = 'cpu'

    if args.use_dp:
        args.use_poison_sampling = True
        args.use_grad_clip = True
        args.delta = args.buffer_size**(-1.1) * args.buffer_throughput

    # Use todays date and time as exp_id
    args.exp_id = datetime.datetime.now().strftime("%m%d-%H%M")




    env_names = ['3m', '8m', '2s3z']
    env_index = 0

        # Log the configs in a json file
    if not os.path.exists('./configs'):
        os.makedirs('./configs')
    with open('./configs/{}_env_{}_number_{}_seed_{}.json'.format(args.algorithm, env_names[env_index], args.exp_id, args.seed), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    runner = QTRAN_Runner(args, env_names[env_index], args.exp_id, args.seed)
    runner.run()
        
