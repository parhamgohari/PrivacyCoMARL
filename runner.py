import argparse
from algos.VDN.VDN_runner import VDN_Runner
from algos.QMIX.QMIX_runner import QMIX_Runner
from algos.IQL.IQL_runner import IQL_Runner
from algos.QTRAN.QTRAN_runner import QTRAN_Runner
import datetime
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QTRAN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--algorithm", type=str, default="VDN", help="IQL, VDN, QMIX, or QTRAN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda_opt", type=float, default=1.0, help="Lambda_opt for QTRAN")
    parser.add_argument("--lambda_nopt", type=float, default=0.1, help="Lambda_nopt for QTRAN")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--q_network_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of Q network")
    parser.add_argument("--encoder_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of encoder")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the mixing network in QMIX")
    parser.add_argument("--hypernet_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of hypernetworks in QMIX")
    parser.add_argument("--hypernet_input_dim", type=int, default=64, help="The dimension of the input of hypernetworks in QMIX")
    parser.add_argument("--qtran_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the mixing network in QTRAN")
    parser.add_argument("--vtran_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the mixing network in VTRAN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="The norm of the gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS")
    parser.add_argument("--use_Adam", type=bool, default=True, help="Whether to use Adam")
    parser.add_argument("--use_SGD", type=bool, default=False, help="Whether to use SGD")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_hard_update", type=bool, default=False, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")
    parser.add_argument("--use_secret_sharing", type=bool, default=True, help="Whether to use secret sharing")
    parser.add_argument("--Q", type=int, default=2**31-1, help="The prime number used in secret sharing")
    parser.add_argument("--precision", type=int, default=5, help="The precision used in secret sharing")
    parser.add_argument("--base", type=int, default=10, help="The base used in secret sharing")
    parser.add_argument("--use_poisson_sampling", type=bool, default=False, help="Whether to use poisson sampling")
    parser.add_argument("--use_dp", type=bool, default=False, help="Whether to use differential privacy")
    parser.add_argument("--noise_multiplier", type=float, default=0.8, help="Noise multiplier")
    parser.add_argument("--delta", type=float, default=None, help="Delta")
    parser.add_argument("--buffer_throughput", type=float, default=1.0, help="Buffer throughput")
    parser.add_argument("--use_anchoring", type=bool, default=False, help="Whether to use anchoring")
    parser.add_argument("--log_and_save", type=bool, default=True, help="Whether to log and save")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    assert (args.use_RMS + args.use_Adam + args.use_SGD) > 0 , "Must choose an optimizer"
    assert (args.use_RMS + args.use_Adam + args.use_SGD) < 2, "Must choose only one optimizer"

    args.device = 'cpu'

    if args.use_dp:
        assert args.use_poisson_sampling == True, "Must use poisson sampling when using differential privacy"
        assert args.use_grad_clip == True, "Must use gradient clip when using differential privacy"
        args.delta = args.buffer_size**(-1.1) * args.buffer_throughput

    # Use todays date and time as exp_id
    args.exp_id = datetime.datetime.now().strftime("%m%d-%H%M")
    # args.exp_id = 'final'

    env_names = ['3m', '3s_vs_4z', '2s3z']
    env_index = 0

    # Log the configs in a json file
    if args.log_and_save:
        if not os.path.exists('./configs'):
            os.makedirs('./configs')
        with open('./configs/{}_env_{}_number_{}_seed_{}.json'.format(args.algorithm, env_names[env_index], args.exp_id, args.seed), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if args.algorithm == 'VDN':
        runner = VDN_Runner(args, env_names[env_index], args.exp_id, args.seed)
    elif args.algorithm == 'QMIX':
        runner = QMIX_Runner(args, env_names[env_index], args.exp_id, args.seed)
    elif args.algorithm == 'IQL':
        runner = IQL_Runner(args, env_names[env_index], args.exp_id, args.seed)
    elif args.algorithm == 'QTRAN':
        runner = QTRAN_Runner(args, env_names[env_index], args.exp_id, args.seed)
    else:
        raise NotImplementedError
    
    runner.run()
