import torch
from tqdm import tqdm
from base.runner_base import Runner_Base

class IQL_Runner(Runner_Base):
    def __init__(self, args, env_name, exp_id, seed):
        super(IQL_Runner, self).__init__(args, env_name, exp_id, seed)

    def run(self):
        verbose = []
        evaluate_num = -1
        pbar = tqdm(total = self.total_steps)
        self.training_initiated = False
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                if self.training_initiated:
                    print('-------------------------------------------------------------------------')
                    print(verbose[-3])
                    print(verbose[-2])
                    print(verbose[-1])
                    print('-------------------------------------------------------------------------')
                if self.args.use_anchoring:
                    self.evaluate_policy(use_anchor_q = True)
                    if self.win_rates[-1] > self.best_win_rate:
                        for agent in self.agents:
                            agent.update_anchor_params()
                        self.best_win_rate = self.win_rates[-1]

                    
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate = False)

            self.total_steps += episode_steps
            pbar.update(episode_steps)
            
            # start training if all agent replay buffer has enough samples
            if min([agent.replay_buffer.current_size for agent in self.agents]) >= (self.args.buffer_size if self.args.use_dp else self.args.batch_size):
                #create an np array of size batch_size * n_agents (sender) * n_agents (receiver)
                self.training_initiated = True
                        
                for agent in self.agents:
                    verbose.append(agent.train(self.total_steps))
                

            if self.args.use_dp:
                self.privacy_budget = {
                    'epsilon': max([agent.accountant.get_epsilon(self.args.delta) for agent in self.agents]) /self.args.buffer_throughput,
                    'delta': self.args.delta /self.args.buffer_throughput
                }


        self.evaluate_policy()
        self.env.close()
        pbar.close()
