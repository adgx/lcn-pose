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
    parser.add_argument('--flip-data', help='train time flip', action='store_true', default=True)
    parser.add_argument('--translate_data', help='train time translate', action='store_true', default=True)
    parser.add_argument("--translation_factor", type=float, default=0.1, help="Factor for translation data augmentation")
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

"""
- H36M: azioni, soggetti, camera -> 
- Humansc3d: soggetti, camera -> 
- MPII: attivita, soggetti, camera -> 

- UNITO: soggetti, camera -> 
"""
def main():
    args = parse_args()

    datareader = data.DataReader()
    gt_trainset_all = datareader.real_read(args.train_set, "train")
    gt_testset_all = datareader.real_read(args.test_set, "test")
    
    gt_trainset = data.get_subset(gt_trainset_all, subset_size=10000, mode="camera")
    gt_testset = data.get_subset(gt_testset_all, subset_size=1000, mode="camera")

    train_data, test_data, train_labels, test_labels = None, None, None, None
    train_data, test_data = datareader.read_2d(gt_trainset, gt_testset, read_confidence=True if args.in_F == 3 else False)
    train_labels, test_labels = datareader.read_3d()

    if args.flip_data:
        train_data_flip = data.flip_data(train_data)
        train_labels_flip = data.flip_data(train_labels)

    if args.translate_data:
        translation_factor = args.translation_factor
        if translation_factor < 0:
            raise ValueError("Translation factor must be non-negative")
        translation = np.random.uniform(-translation_factor, translation_factor)
        train_data_translate = data.translation_data(train_data, translation)
        train_labels_translate = data.translation_data(train_labels, translation)
    
    if args.flip_data:
        train_data = np.concatenate((train_data, train_data_flip), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_flip), axis=0)

    if args.translate_data:
        train_data = np.concatenate((train_data, train_data_translate), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_translate), axis=0)

    if args.output_file is not None:
        if not os.path.exists(os.path.join(ROOT_PATH, 'output', args.output_file)):
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