import argparse

def get_config():
    parser = argparse.ArgumentParser()

    #default
    parser.add_argument("--raw_path", default= '../../data')

    #train
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_fold', default=10, type=int)
    parser.add_argument('--learning_rate',default=1e-2, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--weight_decay',default=5e-4, type=float)
    parser.add_argument('--batch_size', default=10000, type=int)

    #model
    parser.add_argument('--model', default='GraphSAGE')
    parser.add_argument('--hidden', default=16, type=int)
    parser.add_argument('--n_layer', default=3, type=int)
    parser.add_argument('--drop_p', default=0.5, type=float)

    #experiment
    parser.add_argument('--oversampling', action='store_true', default=False)
    parser.add_argument('--weighted_loss', action='store_true', default=False)
    parser.add_argument('--stacking_file', type=str, default=None)

    return parser.parse_args()
