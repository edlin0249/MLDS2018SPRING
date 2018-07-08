def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00015, help='learning rate for training')
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size for replay memory')
    parser.add_argument('--eval_net_update_step', type=int, default=4, help='update step for eval net')
    parser.add_argument('--target_net_update_step', type=int, default=1000, help='update step for target net')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma value for calculating target value')
    parser.add_argument('--save_freq', type=int, default=2000, help='frequence for saving model')
    parser.add_argument('--max_steps', type=int, default=3100000, help='max steps for training')
    parser.add_argument('--log_file', default=None, help='log file for recording vanilla DQN(timesteps v.s. average rewards)')
    parser.add_argument('--vanillaDQN', action='store_true')
    parser.add_argument('--DoubleDQN', action='store_false')
    parser.add_argument('--DuelingDQN', action='store_true')
    parser.add_argument('--model_dqn', default='DoubleDQN_model', help='model path for DQN')
    parser.add_argument('--model_pg', type=str, default='model_pg_4.33')
    return parser
