import torch
from smac.env import StarCraft2Env
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

class Runner_Base(object):
    def __init__(self, args, env_name, exp_id, seed):
        self.args = args
        self.env_name = env_name
        self.exp_id = exp_id
        self.device = args.device
        self.seed = seed
        self.env = StarCraft2Env(map_name=env_name, seed = self.seed)
        self.env_info = self.env.get_env_info()
        self.args.n_agents = self.env_info["n_agents"]
        self.n_agents = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("---------------------------------------")
        print("env_name={}".format(self.env_name))
        print("number of agents={}".format(self.args.n_agents))
        print("obs_dim={}".format(self.args.obs_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        print("algorithm={}".format(self.args.algorithm))
        print("---------------------------------------")
        self.epsilon = self.args.epsilon

        if args.use_dp:
            self.privacy_budget = {
                'epsilon': 0,
                'delta': 0,
            }

        if args.use_anchoring:
            self.best_win_rate = 0.0

        if args.algorithm == 'VDN':
            from algos.VDN.VDN_agent import VDN_Agent as Agent
        elif args.algorithm == 'QMIX':
            from algos.QMIX.QMIX_agent import QMIX_Agent as Agent
        elif args.algorithm == 'IQL':
            from algos.IQL.IQL_agent import IQL_Agent as Agent
        elif args.algorithm == 'QTRAN':
            from algos.QTRAN.QTRAN_agent import QTRAN_Agent as Agent
        else:
            raise NotImplementedError

        self.agents = []
        for id in range(self.n_agents):
            self.update_seed()
            self.agents.append(Agent(self.args, id, self.seed))

        if self.args.log_and_save:
            self.writer = SummaryWriter(log_dir="./log/{}_{}_{}".format(self.args.algorithm, env_name, exp_id))

        self.win_rates = []
        self.total_steps = 0

    def update_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    def run(self):
        evaluate_num = -1
        pbar = tqdm(total = self.total_steps)
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1
            if self.args.use_anchoring:
                    self.evaluate_policy(use_anchor_q = True)
                    if self.win_rates[-1] >= self.best_win_rate:
                        for agent in self.agents:
                            agent.update_anchor_params()
                        self.best_win_rate = self.win_rates[-1]

            _, _, episode_steps = self.run_episode_smac(evaluate = False)

            self.total_steps += episode_steps
            pbar.update(episode_steps)
            

            if min([agent.replay_buffer.current_size for agent in self.agents]) >= (self.args.buffer_size if self.args.use_dp else self.args.batch_size):

                for agent in self.agents:
                    agent.train(self.total_steps)

                self.epsilon = self.epsilon - self.total_steps * self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if self.args.use_dp:
                self.privacy_budget = {
                    'epsilon': max([agent.accountant.get_epsilon(self.args.delta) for agent in self.agents]) /self.args.buffer_throughput,
                    'delta': self.args.delta /self.args.buffer_throughput
                }
        
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
            if self.args.log_and_save:
                self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        else:
            print("total_steps:{} \t anchor_win_rate:{:.2f} \t anchor_evaluate_reward:{:.2f}".format(self.total_steps, win_rate, evaluate_reward))
            if self.args.log_and_save:
                self.writer.add_scalar('anchor_win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        if self.args.use_dp:
            print("privacy budget: epsilon={:.2f} and delta={:.5f}".format(self.privacy_budget['epsilon'], self.privacy_budget['delta']))

    def run_episode_smac(self, evaluate = False, use_anchor_q = False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        for agent in self.agents:
            agent.reset()

        for episode_step in range(self.args.episode_limit):
            for agent in self.agents:
                agent.receive_obs(self.env.get_obs_agent(agent.id))
                agent.receive_avail_a(self.env.get_avail_agent_actions(agent.id))
            epsilon = 0 if evaluate else self.epsilon

            try:
                r, done, info = self.env.step(
                    [agent.choose_action(epsilon, use_anchor_q) for agent in self.agents]
                )
            except:
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
                    agent.store_transition(episode_step = episode_step, dw = dw, r = r)
                
                if self.training_initiated:
                    self.epsilon = max(self.epsilon - self.args.epsilon_decay, self.args.epsilon_min)
            if done:
                break
            
        if not evaluate:
            for agent in self.agents:
                agent.receive_obs(self.env.get_obs_agent(agent.id))
                agent.receive_avail_a(self.env.get_avail_agent_actions(agent.id))
                agent.store_last_step(episode_step = episode_step)
        
        return win_tag, episode_reward, episode_step + 1
    
    def coordinate_learning(self):
        raise NotImplementedError
