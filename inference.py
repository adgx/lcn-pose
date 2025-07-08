import pickle
import tensorflow as tf
import numpy as np
from tools import params_help, data
from network import models_att
import os
import argparse
import pprint

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='inference')

    # optional arguments
    parser.add_argument('--test-indices', help='test idx ', type=str, default="1")
    parser.add_argument('--mask-type', help='mask type', type=str, default='locally_connected', choices=['locally_connected', 'exponential'])
    parser.add_argument('--init-type', help='initialization type ', type=str, default='same', choices=['same', 'ones', 'random'])
    parser.add_argument('--graph', help='index of graphs', type=int, default=0)
    parser.add_argument('--knn', help='expand of neighbourhood', type=int, default=2)
    parser.add_argument('--layers', help='number of layers', type=int, default=3)
    parser.add_argument('--in-joints', help='number of input joints', type=int, default=17)
    parser.add_argument('--out-joints', help='number of output joints', type=int, default=17)
    parser.add_argument('--dropout', help='dropout probability', type=float, default=0.25)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)
    parser.add_argument('--subset', help='Make a subset from the dataset passed', type=int, default=None)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=200)
    parser.add_argument('--checkpoints', help='type of checkpoints', type=str, choices=['final', 'best'])
    parser.add_argument('--batch_size', help='batch size', type=int, default=200)

    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2, choices=[2, 3])
    parser.add_argument('--flip-data', help='train time flip', action='store_true', default=True)
    parser.add_argument('--rotation-data', help='train time rotation', action='store_true', default=True)
    parser.add_argument('--translate_data', help='train time translate', action='store_true', default=True)
    parser.add_argument("--translation_factor", type=float, default=0.1, help="Factor for translation data augmentation")

    parser.add_argument('--train_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    parser.add_argument('--test_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    
    try :
        args = parser.parse_args()
    except Exception as error:
        parser.print_help()
        print(f"Error parsing arguments: {error}")
        print("Please check the arguments and try again.")
        
        raise SystemExit

    return args

def main():
    args = parse_args()
    
    datareader = data.DataReader()
    gt_trainset = datareader.real_read(args.train_set, "train")
    gt_testset = datareader.real_read(args.test_set, "test")
    #Make a subset
    if args.subset is not None:
        gt_trainset = data.get_subset(gt_trainset, subset_size=args.subset, mode="camera")
        gt_testset = data.get_subset(gt_trainset, subset_size=args.subset, mode="camera")

    train_data, test_data, train_labels, test_labels = None, None, None, None
    _, test_data = datareader.read_2d(gt_trainset, gt_testset)
    train_labels, test_labels = datareader.read_3d()

    dataset_copy = test_data.copy()
    labelset_copy = test_labels.copy()
    num_augmentations = 0
    op_ord = {}

    if args.flip_data:
        test_data = np.concatenate((test_data,  data.flip_data(dataset_copy)), axis=0)
        train_labels = np.concatenate((test_labels, data.flip_data(labelset_copy)), axis=0)
        num_augmentations += 1
        op_ord['f'] = num_augmentations

    if args.translate_data:
        translation_factor = args.translation_factor
        if translation_factor < 0:
            raise ValueError("Translation factor must be non-negative")
        translation = np.random.uniform(-translation_factor, translation_factor)
        test_data = np.concatenate((test_data,  data.translation_data(dataset_copy, translation)), axis=0)
        train_labels = np.concatenate((test_labels, data.translation_data(labelset_copy, translation)), axis=0)
        num_augmentations += 1
        op_ord['t'] = num_augmentations

    if args.rotation_data:
        test_data = np.concatenate((test_data, data.rotate_data(dataset_copy)), axis=0)
        train_labels = np.concatenate((test_labels,  data.rotate_data(labelset_copy) ), axis=0)
        num_augmentations += 1
        op_ord['r'] = num_augmentations

    # params
    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    print(pprint.pformat(params))

    network = models_att.cgcnn(**params)
    print("Start with predictions")
    predictions = network.predict(data=test_data, sess=None)  # [N, 17*3]
    print("Predictions done")
    result = {}

    
    if args.flip_data or args.rotation_data or args.translate_data:
        predictions = data.undo(predictions, op_ord, number_actions=num_augmentations, translation=translation)
    result = datareader.denormalize(predictions)

    save_path = os.path.join(ROOT_PATH, 'experiment', params['dir_name'], 'result.pkl')
    f = open(save_path, 'wb')
    pickle.dump({'result': result}, f)
    f.close()

if __name__ == '__main__':
    main()

