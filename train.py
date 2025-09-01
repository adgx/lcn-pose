import numpy as np
from tools import params_help, data
from network import  models_att
import os
import argparse
import pprint
import numpy as np


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='train')

    # optional arguments
    parser.add_argument('--test-indices', help='test idx ', type=str, default="1")
    parser.add_argument('--mask-type', help='mask type', type=str, default='locally_connected', choices=['locally_connected', 'exponential'])
    parser.add_argument('--init-type', help='initialization type ', type=str, default='same', choices=['same'])
    parser.add_argument('--knn', help='expand of neighbourhood', type=int, default=2)
    parser.add_argument('--layers', help='number of layers', type=int, default=3)
    parser.add_argument('--dropout', help='dropout probability', type=float, default=0)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)
    parser.add_argument('--subset', help='Make a subset from the dataset passed', type=int, default=None)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=25)
    parser.add_argument('--batch_size', help='batch size', type=int, default=200)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--regularization', help='regularization factor', type=float, default=None)
    
    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2, choices=[2, 3])
    parser.add_argument('--flip-data', help='train time flip', action='store_true', default=False)
    parser.add_argument('--rotation-data', help='train time rotation', action='store_true', default=False)
    parser.add_argument('--translate_data', help='train time translate', action='store_true', default=False)
    parser.add_argument("--translation_factor", type=float, default=0.1, help="Factor for translation data augmentation")
    parser.add_argument("--rotation_factor", type=float, default=60, help="Factor for rotation in degrees data augmentation")
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint path to resume training from')
    parser.add_argument('--output_file', type=str, default=None, help='Output file to save the model')
    parser.add_argument('--train_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    parser.add_argument('--validation_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    parser.add_argument('--test_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"])
    parser.add_argument('--n_test', help='number of test for random search', type=int, default=1)
    try :
        args = parser.parse_args()
    except:
        parser.print_help()
        raise SystemExit

    return args

def main():
    args = parse_args()

    #read the train and test passed with arguments
    datareader = data.DataReader()
    gt_trainset = datareader.real_read(args.train_set, "train")
    gt_valset = datareader.real_read(args.validation_set, "val")

    mode = ""
    if args.train_set == "h36m":
        mode = "action_camera_subject"
    elif args.train_set == "humansc3d":
        mode = "action_camera_subject"
    elif args.train_set == "mpii":
        mode = "camera"

    #Make a subset
    if args.subset is not None:
        gt_trainset = data.get_subset(gt_trainset, subset_size=args.subset, mode="camera")
        gt_valset = data.get_subset(gt_valset, subset_size=args.subset, mode="camera")

    train_data, val_data, train_labels, val_labels = None, None, None, None
    train_data, val_data = datareader.read_2d(gt_trainset, gt_valset)
    train_labels, val_labels = datareader.read_3d()

    dataset_copy = train_data.copy()
    labelset_copy = train_labels.copy()

    if args.flip_data:
        train_data = np.concatenate((train_data,  data.flip_data(dataset_copy)), axis=0)
        train_labels = np.concatenate((train_labels, data.flip_data(labelset_copy)), axis=0)

    if args.translate_data:
        translation_factor = args.translation_factor
        if translation_factor < 0:
            raise ValueError("Translation factor must be non-negative")
        translation = np.random.uniform(-translation_factor, translation_factor)
        train_data = np.concatenate((train_data,  data.translation_data(dataset_copy, translation)), axis=0)
        train_labels = np.concatenate((train_labels, data.translation_data(labelset_copy, translation)), axis=0)

    if args.rotation_data:
        rotation_factor = args.rotation_factor
        if rotation_factor < 0:
            raise ValueError("Rotation factor must be non-negative")
        rotation = np.random.uniform(-rotation_factor, rotation_factor)
        train_data = np.concatenate((train_data, data.rotate_data(dataset_copy, rotation)), axis=0)
        train_labels = np.concatenate((train_labels,  data.rotate_data(labelset_copy) ), axis=0)

    # params
    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    print(pprint.pformat(params))

    network = models_att.cgcnn(**params)
    try: 
        losses, t_step = network.fit(train_data, train_labels, val_data, val_labels, args.output_file, starting_checkpoint=args.resume_from)
        print(losses)
        print(t_step)
    
    except KeyboardInterrupt:
        print('Training interrupted')
    except Exception as e:
        print('Error during training: ', e)
        raise SystemExit
    
if __name__ == '__main__':
    main()