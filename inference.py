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
    parser.add_argument('--test-indices', help='test idx ', type=str)
    parser.add_argument('--mask-type', help='mask type ', type=str)
    parser.add_argument('--graph', help='index of graphs', type=int, default=0)
    parser.add_argument('--knn', help='expand of neighbourhood', type=int)
    parser.add_argument('--layers', help='number of layers', type=int)
    parser.add_argument('--in-joints', help='number of input joints', type=int, default=17)
    parser.add_argument('--out-joints', help='number of output joints', type=int, default=17)
    parser.add_argument('--dropout', help='keep probability', type=float)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)
    parser.add_argument('--checkpoints', help='type of checkpoints', type=str, choices=['final', 'best'])

    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2)
    parser.add_argument('--flip-data', help='test time flip', action='store_true', default=True)
    parser.add_argument('--augmentation', type=str, default=None, help='Select the kind of data augmantation: flip, rotation, translate', choices=["f", "r", "t"], nargs=3)

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
    gt_trainset_all = datareader.real_read(args.train_set, "train")
    gt_testset_all = datareader.real_read(args.test_set, "test")
    train_data, test_data = datareader.read_2d(gt_trainset_all, gt_testset_all, read_confidence=True if args.in_F == 3 else False)  # [N, 17*2]
    train_labels, test_labels = datareader.read_3d()

    if args.augmentation is not None:
        op_ord = []
        test_data_copy = test_data.copy()
        test_labels_copy =test_labels.copy()
        #flip data 
        if 'f' in args.augmentation:
            #test_data = data.flip_data(test_data)
            test_data = np.concatenate((test_data, data.flip_data(test_data_copy)), axis=0)
            test_labels = np.concatenate((test_labels, data.flip_data(test_labels_copy)), axis=0)
            op_ord.append(data.unflip_data)
        #rotate data
        if 'r' in args.augmentation:
            test_data = np.concatenate((test_data, data.rotate_data(test_data_copy)), axis = 0)
            test_labels = np.concatenate((test_labels, data.rotate_data(test_labels_copy)), axis = 0)
            op_ord.append(data.rotate_data)
        #translate data
        if 't' in args.augmentation:
            test_data = np.concatenate((test_data, data.translation_data(test_data_copy)), axis=0)
            test_labels = np.concatenate((test_labels, data.translation_data(test_labels_copy)), axis=0)
            op_ord.append(data.rotate_data)
    

    # params
    params = params_help.get_params(is_training=False, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    print(pprint.pformat(params))

    network = models_att.cgcnn(**params)
    print("Start with predictions")
    predictions = network.predict(data=test_data, sess=None)  # [N, 17*3]

    #if args.translate_data:
    #    predictions = data.untranslation_data(predictions, translation_factor=params['translation_factor'])  # [N, 17*3]

    #augmented = args.flip_data or args.translate_data
    ## Increase the number of actions if flip_data is True
    #number_action = 0
    #number_action += +1 if args.flip_data else 0
    #number_action += 1 if args.translate_data else 0
    if args.augmentation is not None:
        predictions = data.undo(predictions, op_ord)  # [N, 17*3]
    result = datareader.denormalize(predictions)

    save_path = os.path.join(ROOT_PATH, 'experiment', params['dir_name'], 'result.pkl')
    f = open(save_path, 'wb')
    pickle.dump({'result': result}, f)
    f.close()

if __name__ == '__main__':
    main()

