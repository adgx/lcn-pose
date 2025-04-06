import pickle
import numpy as np
from tools import tools
import os
import prettytable
import argparse
import time

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
THRESHOLD = 10  # mm
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')

    # general
    parser.add_argument('--data-type', help='scale', required=True, choices=['scale'], type=str)
    parser.add_argument('--mode', help='finetuned 2d detection or gt', required=True, choices=['gt', 'dt_ft'], type=str)

    # special
    parser.add_argument('--protocol2', help='whether use Procrustes', action='store_true')
    parser.add_argument('--test-indices', help='test idx list to eval', required=True, type=str)
    parser.add_argument('--per-joint',  help='joint-wise evaluation', action='store_true')
    parser.add_argument('--filename', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    try :
        args = parser.parse_args()
    except:
        parser.print_help()
        raise SystemExit

    return args


def _eval(test_name, dataitem_gt, commd, mode):
    result_path = os.path.join(ROOT_PATH, 'experiment', test_name, 'result_%s.pkl' % mode)

    with open(result_path, 'rb') as f:
        preds = pickle.load(f)['result']  # [N, 17, 3]
    preds = np.reshape(preds, (-1, 17, 3))

    assert len(preds) == len(dataitem_gt)

    results = []
    best_frame = []
    best_frame_id = []
    for idx, pred in enumerate(preds):
        pred = tools.image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
            camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
            root_depth=dataitem_gt[idx]['root_depth'])
        gt = dataitem_gt[idx]['joint_3d_camera']
        if 'protocol2' in commd:
            pred = tools.align_to_gt(pose=pred, pose_gt=gt)
        error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
        results.append(error_per_joint)
        if idx % 10000 == 0:
            print('step:%d' % idx + '-' * 20)
            print(np.mean(error_per_joint))
    results = np.array(results)  # [N ,17]

    if 'action' in commd:
        final_result = []
        final_results_pck = []
        action_index_dict = {}
        for i in range(2, 17):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(dataitem_gt):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 17):
            final_result.append(np.mean(results[action_index_dict[i]])) # media degli errori dell'azione
            
            arr = np.array(results[action_index_dict[i]])
            if len(arr < THRESHOLD) == 0:
                final_results_pck.append(0)
            else:
                final_results_pck.append((len(arr[arr < THRESHOLD]) / (len(arr) * arr.shape[1])) * 100)


        error = np.mean(np.array(final_result))
        final_result.append(error)

        final_results_pck.append(np.mean(np.array(final_results_pck)))
    elif 'joint' in commd:
        error = np.mean(results, axis=0)  # [17]
        final_result = error.tolist() + [np.mean(error)]
    else:
        assert 0, 'not implemented commd'
    
    return final_result, final_results_pck


def eval(commd, test_indices, mode='dt', pkl=""):
    err_dict = {}

    print('loading dataset')

    DATAITEM_GT_PATH = os.path.join(ROOT_PATH, "dataset/" + pkl + "_test.pkl")
    with open(DATAITEM_GT_PATH, 'rb') as f:
        dataitem_gt = pickle.load(f)

    # eval each trial
    for i in test_indices:
        test_name = 'test%d' % i
        err_dict[test_name], final_results_pck = _eval(test_name, dataitem_gt, commd, mode)



if __name__ == '__main__':

    args = parse_args()
    test_indices = sorted([int(i) for i in args.test_indices.split(',')])
    commd = args.data_type
    if args.per_joint:
        commd += '_joint'
    else:
        commd += '_action'
    if args.protocol2:
        commd += '_protocol2'

    eval(commd=commd, test_indices=test_indices, mode=args.mode, pkl=args.filename)

