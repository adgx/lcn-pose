import os
import os.path as osp
import json 
import numpy as np
import parse
import argparse
import pickle


def find_train_val_dirs(dataset_root_dir):
    trainsubs = ['s01', 's02', 's03', 's06']
    train_dirs, val_dirs = [], []
    allsets = os.listdir(dataset_root_dir)
    for dset in allsets:
        if osp.isdir(osp.join(dataset_root_dir, dset)):
            if dset[:4] in trainsubs:
                train_dirs.append(dset)
            else:
                val_dirs.append(dset)
    train_dirs.sort()
    val_dirs.sort()
    return train_dirs, val_dirs

def infer_meta_from_name(datadir):
    format_str = 's_{}_ca_{}'
    res = parse.parse(format_str, datadir)
    meta = {
        'subject': int(res[0]),
        'camera': int(res[1])
    }
    return meta

def _retrieve_camera(camera, subject, cameraidx):
    R, T = camera['extrinsics']
    f, c, k, p = camera['intrinsics_w_distortion'] #what about intrinsics_wo_distortion?
    camera = {}
    camera['R'] = R
    camera['T'] = T
    camera['fx'] = f[0]
    camera['fy'] = f[1]
    camera['cx'] = c[0]
    camera['cy'] = c[1]
    camera['k'] = k
    camera['p'] = p
    #camera['name'] = name
    return camera

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame

#cams it is a np array
def load_db(dataset_root_dir, dset, vid, cams, rootIdx=0):
    annojointsfile = os.path.join(dataset_root_dir, dset, '001.json')
    with open(annojointsfile, 'r') as f:
        joints_data = json.load(f)
    
    joints_3d_cam = np.array(joints_data['joints3d_25'])
    numimgs = joints_3d_cam.shape[0]
    joints_3d_cam = np.reshape(np.transpose(joints_3d_cam, (0, 1, 2)), (numimgs, -1, 3))
    meta = infer_meta_from_name(dset)
    cam = _retrieve_camera(cams, meta['subject'], meta['camera'])


    dataset = []
    for i in range(numimgs):
        image = 's_{:02}_ca_{:02}_{:06}.jpg'.format(
            meta['subject'], meta['camera'], i + 1)
        image = os.path.join(dset, image)
        joint_3d_cam = joints_3d_cam[i]#obtain the all joints position for the frame
        box = _infer_box(joint_3d_cam, cam, rootIdx)#obtain info about bounding box
        joint_3d_image = camera_to_image_frame(joint_3d_cam, box, cam, rootIdx)
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) 
        scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
        dataitem = {
            'videoid': vid,
            'cameraid': meta['camera'],
            'camera_param': cam,
            'imageid': i,
            'image_path': image,
            'joint_3d_image': joint_3d_image,
            'joint_3d_camera': joint_3d_cam,
            'center': center,
            'scale': scale,
            'box': box,
            'subject': meta['subject'],
            'root_depth': joint_3d_cam[rootIdx, 2]
        }

        dataset.append(dataitem)
    
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir')
    args = parser.parse_args()

    cams = os.path.join(args.dataset_root_dir, 's01', 'camera_parameters', '65906101', '001.json')

    train_dirs, val_dirs = find_train_val_dirs(args.dataset_root_dir)
    train_val_datasets = [train_dirs, val_dirs]
    dbs = []
    video_count = 0
    for dataset in train_val_datasets:
        db = []
        for video in dataset:
            if np.mod(video_count, 1) == 0:
                print('Process {}: {}'.format(video_count, video))

            data = load_db(args.dataset_root_dir, video, video_count, cams)
            db.extend(data)
            video_count += 1
        dbs.append(db)

    datasets = {'train': dbs[0], 'validation': dbs[1]}

    with open(args.dataset_root_dir + 'h36m_train.pkl', 'wb') as f:
        pickle.dump(datasets['train'], f)

    with open(args.dataset_root_dir + 'h36m_test.pkl', 'wb') as f:
        pickle.dump(datasets['validation'], f)
