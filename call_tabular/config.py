import argparse

def get_config():
    parser = argparse.ArgumentParser()

    #default
    parser.add_argument("--raw_path", default= '../data')

    #train
    parser.add_argument('--model', default='DNN')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_fold', default=10, type=int)
    parser.add_argument('--learning_rate',default=1e-2, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--weight_decay',default=5e-4, type=float)

    #model
    parser.add_argument('--hidden', default=16, type=int)
    parser.add_argument('--drop_p', default=0.5, type=float)

    return parser.parse_args()
