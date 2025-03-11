import os
import os.path as osp
import json 
import numpy as np
import parse
import argparse
import pickle
import sys

from data_preparation import convert_humansc3d_mp4_to_image
from gen_val_set import generate_validation_set

def load_cams_data(dataset_root_dir, subset, subj_name, camera_param):
    path_cameras = os.path.join(dataset_root_dir, subset, subj_name, camera_param)
    #check the path
    if not osp.isdir(path_cameras):
        print(f'Error path isn\'t valid: {path_cameras}')

    cams_data = {}

    for camera_view in camera_ids:
        path_camera = os.path.join(path_cameras, camera_view, '001.json')   
        with open(path_camera, 'r') as cam_json:
            cams_data[camera_view] = json.load(cam_json)
    return cams_data

def find_dirs(dataset_root_dir, subset, subj_names):
    dirs = []

    if not osp.isdir(osp.join(dataset_root_dir, subset)):
        return dirs
    
    allsubjsets = os.listdir(osp.join(dataset_root_dir, subset))
    
    for subj in allsubjsets:
        if subj[:3] in subj_names:
            dirs.append(subj)
    dirs.sort()
    return dirs

#the action is the name of the video for the dataset humansc3d
#humansc3d doesn't have the subaction information so we apply a placeholder
def infer_meta_from_name(subj_video, action, cam_id):
    meta = {
        'subject': int(subj_video[1:]),
        'action': int(action[:3]),
        'subaction': int(0),
        'camera': int(cam_id)
    }
    return meta

def _retrieve_camera(cams, subject, cameraidx):
    R, T = cams[str(cameraidx)]['extrinsics'].values()
    f, c, k, p = cams[str(cameraidx)]['intrinsics_w_distortion'].values() #what about intrinsics_wo_distortion?
    camera = {}
    camera['R'] = R
    camera['T'] = T[0]
    camera['fx'] = f[0][0]
    camera['fy'] = f[0][1]
    camera['cx'] = c[0][0]
    camera['cy'] = c[0][1]
    camera['k'] = k[0]
    camera['p'] = p[0]
    camera['name'] = [subject, cameraidx] #hypothesis
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
def load_db(dataset_root_dir, dset, joints_dir, images_dir, cams, rootIdx=0):
    
    videos_joints = os.listdir(os.path.join(dataset_root_dir, joints_dir))
    dataset = []
    
    for video_joint in videos_joints:
        print(f'load: {dset}:video:{video_joint[:3]}')
        annojointsfile = os.path.join(dataset_root_dir,  joints_dir, video_joint)
        with open(annojointsfile, 'r') as f:
            joints_data = json.load(f)
        joints_3d_cam = np.array(joints_data['joints3d_25'])
        numimgs = joints_3d_cam.shape[0]
        joints_3d_cam = np.reshape(np.transpose(joints_3d_cam, (0, 1, 2)), (numimgs, -1, 3))

        for camera_id in cams:
            meta = infer_meta_from_name(dset, video_joint, camera_id)
            cam = _retrieve_camera(cams, meta['subject'], meta['camera'])#handle multicamera pov

            
            for i in range(numimgs):
                image = os.path.join(dataset_root_dir, images_dir, camera_id, video_joint[:3], 'frame_'+str(i).zfill(4)+'.jpeg')
                joint_3d_cam = joints_3d_cam[i]#obtain the all joints position for the frame
                box = _infer_box(joint_3d_cam, cam, rootIdx)#obtain info about bounding box
                joint_3d_image = camera_to_image_frame(joint_3d_cam, box, cam, rootIdx)
                center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) 
                scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                dataitem = {
                    'videoid': video_joint[:3],
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
    parser = argparse.ArgumentParser("Script to generate the pkl format from the humansc3d")
    parser.add_argument("-t", "--train", action="store_true", help="Create the pkl for the training set")
    parser.add_argument("-v", "--validation", action="store_true", help="Create the pkl for the validation set")
    parser.add_argument("-m", "--image", action="store_true", help="Convert dataset mp4 videos to images")
    parser.add_argument("-g", "--gen", action="store_true", help="Generate validation set from training set")
    args = parser.parse_args()
    
    #debug stuff 
    current_dir = os.getcwd()
    print("Current dir:", current_dir)
    
    #information to retrieve the dataset information
    dataset_name = 'humansc3d'
    dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
    subset_type = ['train', 'test']
    subj_name_train = ['s01', 's02', 's03', 's06']
    subj_name_val = ['s04', 's05']
    joints_dir = 'joints3d_25'
    videos_dir = 'videos'
    images_dir = 'images'
    camera_param = 'camera_parameters'
    camera_ids = ['50591643', '58860488', '60457274', '65906101']

    # loading cameras data
    # we assuming that camera parameters doesn't change in the time and keep the same between train and validation set
    cams = load_cams_data(dataset_root_dir, subset_type[0], subj_name_train[0], camera_param)

    train_dirs = []
    val_dirs = []
    #allow to find the train and validation set
    if args.train:
        train_dirs = find_dirs(dataset_root_dir, subset_type[0], subj_name_train)

    if args.validation:
        val_dirs = find_dirs(dataset_root_dir, subset_type[1], subj_name_val)

    train_val_datasets = [train_dirs, val_dirs]
    dbs = []
    video_count = 0
    idx_subset_type = 0 
    for dataset in train_val_datasets:
        db = []
        for subj_video in dataset:
            base_path = os.path.join(dataset_root_dir, subset_type[idx_subset_type], subj_video)
            if np.mod(video_count, 1) == 0:
                print('Process {}: {}'.format(video_count, subj_video))

            if args.image and dataset_name == 'humansc3d':  
                convert_humansc3d_mp4_to_image(base_path,  videos_dir, images_dir, camera_ids)

            data = load_db(base_path, subj_video, joints_dir, images_dir, cams)
            db.extend(data)
            video_count += 1
        dbs.append(db)
    idx_subset_type += 1

    datasets = {'train': dbs[0], 'validation': dbs[1]}

    if args.train:
        with open(os.path.join(dataset_root_dir,'humansc3d_train.pkl'), 'wb') as f:
            pickle.dump(datasets['train'], f)
    
    if args.validation:
        with open(os.path.join(dataset_root_dir, 'humansc3d_val.pkl'), 'wb') as f:
            pickle.dump(datasets['validation'], f)
