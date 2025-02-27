# 1. Argument parser (todo: dataclass)

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Train a model on EUPP')

    parser.add_argument('--loss', type=str, default='CRPS',
                        choices=['CRPS',  'CRPSTRUNC'],
                        help='Loss function for training (default: CRPS)')

    parser.add_argument('--seed', type=int, default=16,
                        help='Torch Seed (default: 16)')


    parser.add_argument('--ens-num', type=int, default=11,
                      )

    parser.add_argument('--data-path', type=str, default='./',
                        help='The path for both EUPP and ERA5 datasets (default: ./)')

    parser.add_argument('--target-var', type=str, default='t2m',
                        choices=['w100','w10', 't2m'],
                        help='Target variable for prediction (default: t2m)')

    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--lr', '-lr', default=1e-2, type=float,
                        help='Learning rate (default: 1e-2)')

    parser.add_argument('--epochs', type=int , default=10,
                        help='Epochs (default: 10)')

    parser.add_argument('--batch-size', '-b', type=int , default=32,
                        help='Batch size (default: 32)')
    
    # Add arguments for mlp_mult, projection_channels, and num_blocks
    parser.add_argument('--mlp_mult', type=int, default=8,
                        help='MLP multiplier (default: 8)')
    
    parser.add_argument('--projection_channels', type=int, default=128,
                        help='Projection channels (default: 128)')

    parser.add_argument('--num_blocks', type=int, default=5,
                        help='Number of transformer blocks (default: 5)')
    
    parser.add_argument('--nheads', type=int, default=8,
                        help='Number of heads (default: 5)')
        parser.add_argument('--num_predictors', type=int, default=12,
                        help='Number of predictors (default: 12)')

    args = parser.parse_args()
    return args

