import torch
from tqdm import tqdm

from base.runner_base import Runner_Base

class QMIX_Runner(Runner_Base):
    def __init__(self, args, env_name, exp_id, seed):
        super(QMIX_Runner, self).__init__(args, env_name, exp_id, seed)

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
                skip_iteration = False
                for agent in self.agents:
                    agent.initiate_peer2peer_messaging(self.seed)
                    if agent.empty_batch:
                        skip_iteration = True
                        break
                self.update_seed()
                if skip_iteration:
                    continue
                

                # Compute the first layer of q_mix
                for agent in self.agents:
                    agent.peer2peer_messaging_for_computing_qmix_pt1(mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_qmix_pt1(mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_qmix_pt1(mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_qmix_pt1(mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_qmix_pt1(mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))

                # Compute the second layer of q_mix
                for agent in self.agents:
                    agent.peer2peer_messaging_for_computing_qmix_pt2(mode = '1. compute message')
                local_sums = []
                for receiver_agent in self.agents:
                    for sender_agent in self.agents:
                        receiver_agent.peer2peer_messaging_for_computing_qmix_pt2(mode = '2. receive message', sender_id = sender_agent.id, sender_message = sender_agent.secret_shares[:,:,:,receiver_agent.id])
                    local_sums.append(receiver_agent.peer2peer_messaging_for_computing_qmix_pt2(mode = '3. compute sum'))
                local_sums = torch.stack(local_sums)

                for agent in self.agents:
                    if self.args.use_secret_sharing:
                        agent.peer2peer_messaging_for_computing_qmix_pt2(mode = '4. receive the sum of local sums', sender_message = local_sums)
                    else:
                        agent.peer2peer_messaging_for_computing_qmix_pt2(mode = '4. receive the sum of local sums', sender_message = torch.sum(local_sums, dim = 0))
 
                # train
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
