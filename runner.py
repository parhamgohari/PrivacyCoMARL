from smac.env import StarCraft2Env
import numpy as np
from agent import AgentVDN, AgentQMIX, AgentBase
import argparse


comarl_algorithm_list = ['VDN', 'QMIX']
privacy_building_block_list = ['centralized', 'decentralized', 'decentralized_SMC', 'decentralized_DP', 'Decentralized_SMC_DPSGD']
map_name_list = ['8m']

# Takes the joint observation and returns a specific agent's observation
def local_observation_mapping(obs, agent_id):
    return obs[agent_id]

# Picks a set of indices for the minibatch elements in every iteration
def minibatch_indices(buffer_size, batch_size):   
    return np.random.choice(buffer_size, batch_size, replace = False)

def decentralized_runner(args, env, comarl_algorithm = 'VDN', n_episodes = 10, minibatch_size = 128, target_update_frequency = 1000):
    
    env_info = env.get_env_info()
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]  # The number of agents
    args.obs_dim = env_info["obs_shape"]  # The dimensions of an agent's observation space
    args.state_dim = env_info["state_shape"]  # The dimensions of global state space
    args.action_dim = env_info["n_actions"]  # The dimensions of an agent's action space
    args.episode_limit = env_info["episode_limit"]  # Maximum number of steps per episode
    print("number of agents={}".format(n_agents))
    print("obs_dim={}".format(args.obs_dim))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("episode_limit={}".format(args.episode_limit))

    # agents = []
    # for agent_id in range(n_agents):
    #     if comarl_algorithm == 'VDN':
    #         agents.append(AgentVDN(args))
    #     elif comarl_algorithm == 'QMIX':
    #         agents.append(AgentQMIX(args))
    #     else:
    #         print('Co-MARL algorithm not supported. Try VDN or QMIX. Current algorithm: {}'.format(comarl_algorithm))

    args.use_history_encoding = False

    agents = [AgentBase(args) for _ in range(n_agents)]

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        transition_id = 0

        if args.use_rnn == True:
            for agent in agents:
                agent.eval_Q_net.reset_rnn_hidden()

        while not terminated:
            obs = env.get_obs()
            # state = env.get_state()
            # env.render()  # Uncomment for rendering
            
            joint_action = []
            # Agents make observations and choose actions
            for agent_id, agent in enumerate(agents):
                local_obs = local_observation_mapping(obs, agent_id)
                agent.receive_observation(local_obs)
                available_actions = env.get_avail_agent_actions(agent_id)
                action, q_value = agent.choose_action(local_obs, available_actions, args.epsilon)
                joint_action.append(action)
            
            # Team receives rewards and environment transitions
            team_reward, terminated, _ = env.step(joint_action)
            episode_reward += team_reward

            next_obs = env.get_obs()

            """for every agent, save (last_action, last_local_obs), team_reward, terminated, action, (action, local_obs) to its buffer"""
            for agent_id, agent in enumerate(agents):

                agent.receive_reward(team_reward)
                agent.receive_terminated(terminated)
                agent.receive_next_observation(next_obs[agent_id])
                agent.update_buffer(transition_id)

            for sender_agent in agents:
                for recipient_agent in agents:
                    message = sender_agent.send_message(method = 'decentralized')
                    recipient_agent.receive_message(message, method = 'decentralized')
            
            # Agents perform local updates
            if transition_id >= minibatch_size:
                for agent in agents:
                    agent.update(minibatch_indices())
                    if transition_id % target_update_frequency == 0:
                        agent.update_target_network()

        print("Total reward in episode {} = {}".format(e, episode_reward))
        transition_id += 1

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
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
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
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


    env = StarCraft2Env(map_name = '8m')
    decentralized_runner(args, env)