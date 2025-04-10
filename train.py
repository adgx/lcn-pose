import pickle
import tensorflow as tf
import numpy as np
from tools import tools, params_help, data
from network import  models_att # models_attr2 as
import os
import argparse
import pprint
import numpy as np


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='train')

    # optional arguments
    parser.add_argument('--test-indices', help='test idx ', type=str)
    parser.add_argument('--mask-type', help='mask type ', type=str)
    parser.add_argument('--graph', help='index of graphs', type=int, default=0)
    parser.add_argument('--knn', help='expand of neighbourhood', type=int)
    parser.add_argument('--layers', help='number of layers', type=int)
    parser.add_argument('--in-joints', help='number of input joints', type=int, default=17)
    parser.add_argument('--out-joints', help='number of output joints', type=int, default=17)
    parser.add_argument('--dropout', help='dropout probability', type=float)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)

    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2, choices=[2, 3])
    parser.add_argument('--flip-data', help='train time flip', action='store_true')
    parser.add_argument('--output_file', type=str, default=None, help='Output file where save the informations pf the process')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint path to resume training from')
    parser.add_argument('--train_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    parser.add_argument('--test_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)

    try :
        args = parser.parse_args()
    except:
        parser.print_help()
        raise SystemExit

    return args


def main():
    args = parse_args()

    datareader = data.DataReader()
    gt_trainset_all = datareader.real_read(args.train, "train")
    gt_testset_all = datareader.real_read(args.test, "test")
    
    mask = np.random.randint(1, 2, len(gt_trainset_all)).tolist()
    gt_trainset = [val for val, mask in zip(gt_trainset_all, mask) if mask == 1]

    mask = np.random.randint(1, 2, len(gt_trainset_all)).tolist()
    gt_testset = [val for val, mask in zip(gt_testset_all, mask) if mask == 1]

    train_data, test_data = datareader.read_2d(gt_trainset, gt_testset, read_confidence=True if args.in_F == 3 else False)
    train_labels, test_labels = datareader.read_3d()

    if args.flip_data:
        # only work for scale 
        train_data = data.flip_data(train_data)
        train_labels = data.flip_data(train_labels)

    args.output_file = os.path.join(ROOT_PATH, 'output', 'output.txt')
    if args.output_file is not None:
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))

    # params
    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    print(pprint.pformat(params))

    network = models_att.cgcnn(**params)
    try: 
        losses, t_step = network.fit(train_data, train_labels, test_data, test_labels, args.output_file, starting_checkpoint=args.resume_from)
        print(losses)
        print(t_step)
    
    except KeyboardInterrupt:
        print('Training interrupted')
    except Exception as e:
        print('Error during training: ', e)
        raise SystemExit
    
if __name__ == '__main__':
    main()

