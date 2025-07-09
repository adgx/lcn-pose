import numpy as np
import scipy
import os, sys
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_DIR)
import filter_hub

def get_neighbour_matrix_by_hand(neighbour_dict, knn=1):
    """
    Generate a neighbour matrix from a dictionary of neighbours.
    """
    assert len(neighbour_dict) == 17
    neighbour_matrix = np.zeros((17, 17), dtype=np.float32)
    for idx in range(len(neighbour_dict)):
        neigbour = [idx] + neighbour_dict[idx]
        neighbour_matrix[idx, neigbour] = 1
    if knn >= 2:
        neighbour_matrix = np.linalg.matrix_power(neighbour_matrix, knn)
        neighbour_matrix = np.array(neighbour_matrix!=0, dtype=np.float32)
    return neighbour_matrix


def update_parameters(args, params):
    if args.test_indices:
        params['dir_name'] = 'test' + args.test_indices + '/'
    if args.knn:
        params['neighbour_matrix'] = get_neighbour_matrix_by_hand(filter_hub.neighbour_dict_set[0], knn=args.knn)
    if args.layers is not None:
        params['num_layers'] = args.layers
    if args.dropout is not None:
        params['dropout'] = args.dropout
    if hasattr(args, 'channels') and args.channels:
        params['F'] = args.channels
    if hasattr(args, 'checkpoints') and args.checkpoints:
        params['checkpoints'] = args.checkpoints
    if args.mask_type:
        params['mask_type'] = args.mask_type
    if args.init_type:
        params['init_type'] = args.init_type
    if args.epochs:
        params['num_epochs'] = args.epochs
    if args.batch_size:
        params['batch_size'] = args.batch_size
    if hasattr(args, "learning_rate"):
        params['learning_rate'] = args.learning_rate
    if hasattr(args, "regularization"):
        params['regularization'] = args.regularization


def get_params(is_training, gt_dataset):

    params = {}
    params['dir_name'] = 'test1/'
    params['num_epochs'] = 200
    params['batch_size'] = 200
    # decay_strategy: lr * decay_rate ^ (epoch_num)
    params['decay_type'] = 'exp'  # 'step', 
    params['decay_params'] = {'decay_steps': 32000, 'decay_rate':0.96}  # param for exponential decay optimizer
    #params['decay_params'].update({'boundaries': [250000, 500000, 1000000, 1350000], 'lr_values': [1e-3, 7e-4, 4e-4, 2e-4, 1e-4]})  # param for step optimizer
    #params['eval_frequency'] = int(len(gt_dataset) / params['batch_size'])  # eval, summ & save after each epoch

    params['F'] = 64
    params['mask_type'] = 'locally_connected'
    params['init_type'] = 'random'  # same, ones, random; only used when learnable
    params['neighbour_matrix'] = get_neighbour_matrix_by_hand(filter_hub.neighbour_dict_set[0], knn=3)

    params['in_joints'] = 17
    params['out_joints'] = 17
    params['num_layers'] = 3
    params['in_F'] = 2
    params['residual'] = True
    params['max_norm'] = True
    params['batch_norm'] = True

    params['regularization'] = 0  # 5e-4, 0.0
    params['dropout'] = 0.25 if is_training else 0  # drop prob
    params['learning_rate'] = 1e-3
    params['checkpoints'] = 'final'
    params['is_training'] = is_training

    return params