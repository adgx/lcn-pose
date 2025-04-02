import os
import os.path as osp
import scipy.io as sio
import json 
import numpy as np
import parse
import argparse
import pickle
import sys

from data_preparation import convert_humansc3d_mp4_to_image
from gen_val_set import generate_validation_set
def load_cams_data(dataset_root_dir, subset, subj_name, camera_param):
    dataset_name = os.path.basename(dataset_root_dir)
    cams_data = {}
    if dataset_name == 'humansc3d':
        cams_data = load_cams_data_humansc3d(dataset_root_dir, subset, subj_name, camera_param)    
    elif dataset_name == 'mpi_inf_3dhp':
        cams_data = load_cams_data_mpii(dataset_root_dir, subset, subj_name)

    return cams_data

    
def load_cams_data_humansc3d(dataset_root_dir, subset, subj_name, camera_param):
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

#note: for mpii dataset
#-camera:
#example of Camera calibaration parameters
#name          0
#  sensor      10 10
#  size        2048 2048
#  animated    0
#  intrinsic   1497.693 0 1024.704 0 0 1497.103 1051.394 0 0 0 1 0 0 0 0 1 
#  extrinsic   0.9650164 0.00488022 0.262144 -562.8666 -0.004488356 -0.9993728 0.0351275 1398.138 0.262151 -0.03507521 -0.9643893 3852.623 0 0 0 1 
#  radial      0

#Intrinsics:
#The “intrinsic” line is given as a 4×4 matrix in row-major order:

#[1497.693   0         1024.704   0
# 0         1497.103   1051.394   0
# 0         0          1          0
# 0         0          0          1]

#Here, the camera’s focal lengths and principal point are derived from the top‑left 3×3 block:
#
#    Focal lengths (f): fx = 1497.693, fy = 1497.103
#
#    Principal point (c): cx = 1024.704, cy = 1051.394

#Extrinsics:
#The “extrinsic” line represents a 4×4 matrix:
#
#[ 0.9650164    0.00488022   0.262144    -562.8666
# -0.004488356 -0.9993728   0.0351275   1398.138
#  0.262151   -0.03507521  -0.9643893   3852.623
#  0           0           0           1 ]
#
#The top‑left 3×3 block is the rotation matrix R and the right‑most column (first three entries) is the translation vector T

#The “radial” parameter is provided as “0”. Since no tangential parameters are specified, we assume zero distortion. In many 
# calibration formats the radial distortion is represented with three coefficients (k1, k2, k3) and tangential with two (p1, p2). Here, we set:
#
#    k: [0, 0, 0]
#
#    p: [0, 0]

def load_cams_data_mpii(dataset_root_dir, subset, subj_name):
    path_cameras = os.path.join(dataset_root_dir, subset, subj_name)
    #check the path
    if not osp.isdir(path_cameras):
        print(f'Error path isn\'t valid: {path_cameras}')

    cams_data = {}

    path_camera = os.path.join(path_cameras, 'Seq1', 'camera.calibration')   
    with open(path_camera, 'r') as cam_cal:
        next(cam_cal)
        for line in cam_cal:
            tokens = line.split()
            key, *values = tokens

            if key == 'name':
                name = values[0]  # Store as string, not list
                cam_data = {}
            elif key in ('sensor', 'animated', 'size'):
                continue
            elif key == 'intrinsic':
                cam_data['intrinsics_w_distortion'] = {
                    'f': [float(values[0]), float(values[5])],
                    'c': [float(values[2]), float(values[6])],
                    'k': [0.0, 0.0, 0.0],  # Default values
                    'p': [0.0, 0.0]
                }
            elif key == 'extrinsic':
                cam_data['extrinsic'] = {
                    'R': [
                        list(map(float, values[:3])),
                        list(map(float, values[4:7])),
                        list(map(float, values[8:11]))
                    ],
                    'T': list(map(float, [values[3], values[7], values[11]]))
                }
            elif key == 'radial':  # Assuming 'radial' marks the end of a camera block
                cams_data[name] = cam_data

    return cams_data

def find_dirs(dataset_root_dir, subset, subj_names):
    dirs = []

    if not osp.isdir(osp.join(dataset_root_dir, subset)):
        return dirs
    
    allsubjsets = os.listdir(osp.join(dataset_root_dir, subset))
    
    for subj in allsubjsets:
        if subj in subj_names:
            dirs.append(subj)
    dirs.sort()
    return dirs

#the action is the name of the video for the dataset humansc3d
#humansc3d doesn't have the subaction information so we apply a placeholder
def infer_meta_from_name(subj_video, action, cam_id):
    meta = {
        'subject': int(subj_video[1:3]),
        'action': int(action[:3]),
        'subaction': int(0),
        'camera': int(cam_id)
    }
    return meta


#A raw way to get information from dirs
def infer_meta_from_name_mpii(subj_video, action, cam_id):
    meta = {
        'subject': int(subj_video[1]),
        'action': int(action[3]),
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
    #camera['name'] = [subject, cameraidx] #hypothesis
    camera['name'] = str(cameraidx)
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

#Pw​=(R^−1)×(Pc​−T)
#to obtain all Pw, make a Pc matrix with all posistion joints like: Pc0^T...Pcn^T; 
#and make A Matrix of repete T^T n times.
#R^-1 = R^T
#dset = subject dir
#    case 'relevant' %Human3.6m compatible joint set in Our order
#        joint_idx = [8 = head_top, 6 = neck, 15 = right_shoulder, 16 = right_elbow, 17 = right_wrist, 
#                     10 = left_shoulder, 11 = left_elbow, 12 = left_wrist, 24 = right_hip, 25 = right_knee,
#                     26 = right_ankle, 19 = left_hip, 20 = left_knee, 21 = left_ankle, 5 = pelvis, 4 = spine, 7 = head];
#So to find the corrispondance between joint_idx mpii and h3.6m
#reording indices mpii for h3.6m: [5, 24, 25, 26, 19, 20, 21, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17]
def load_db_mpii(dataset_root_dir, dset, cams, rootIdx=0):
    seq_video_dirs = os.listdir(os.path.join(dataset_root_dir))
    dataset = []
    
    for seq_video_anno in seq_video_dirs:
        print(f'load {dset} {seq_video_anno}')
        annofile = os.path.join(dataset_root_dir, seq_video_anno, 'annot.mat')
        
        if not os.path.exists(annofile):
            print(f"{annofile}: file does not exist")
            
        anno = sio.loadmat(annofile)
        array_joint = anno['univ_annot3'][0][0]
        #edit array of joints positions
        print(f'Formatting joints data ')
        #get the Traslation vector and rotation matrix of the cam 0
        joints_3d_cam  = []
        R_t = np.array(cams['0']['extrinsic']['R']).transpose()
        T = np.array(cams['0']['extrinsic']['T'])
        T = T.reshape(-1, 1)
        T_matrix = np.tile(T, (1, 17))
        for frame_joints_pos in array_joint:
            frame_joints_pos_flat = frame_joints_pos.flatten()
            reshaped_joints_pos = frame_joints_pos_flat.reshape(3, -1, order='F')
            relevant_joints_pos = reshaped_joints_pos[:, [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]]
            joints_pos_w = np.matmul(R_t, (relevant_joints_pos - T_matrix)).transpose()
            joints_3d_cam.append(joints_pos_w)
        joints_3d_cam = np.array(joints_3d_cam)
        numimgs = joints_3d_cam.shape[0]
        
        for camera_id in cams:
            meta = infer_meta_from_name_mpii(dset, seq_video_anno, camera_id)
            cam = _retrieve_camera(cams, meta['subject'], meta['camera'])#handle multicamera pov

            
            for i in range(numimgs):
                image = os.path.join(dataset_root_dir, images_dir, camera_id, video_joint[:3], 'frame_'+str(i).zfill(4)+'.jpeg')
                joint_3d_cam = joints_3d_cam[i, :17, :]#obtain the all joints position for the frame
                box = _infer_box(joint_3d_cam, cam, rootIdx)#obtain info about bounding box
                joint_3d_image = camera_to_image_frame(joint_3d_cam, box, cam, rootIdx)
                center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) 
                scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                dataitem = {
                    'videoid': seq_video_anno[3],
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
                    'action': meta['action'],
                    'root_depth': joint_3d_cam[rootIdx, 2]
                }

                dataset.append(dataitem)
    
    return dataset    

    

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
                joint_3d_cam = joints_3d_cam[i, :17, :]#obtain the all joints position for the frame
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
                    'action': meta['action'],
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
    parser.add_argument("-d", '--dataset', type=str, default="humansc3d", help='Filename of the dataset', choices=["mpii", "humansc3d"])
    args = parser.parse_args()
    
    #debug stuff 
    current_dir = os.getcwd()
    print("Current dir:", current_dir)
    
    #information to retrieve the dataset information

    if args.dataset == "humansc3d":
        dataset_name = args.dataset
        dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
        subset_type = ['train', 'test']
        subj_name_train = ['s01', 's02', 's03', 's06']
        subj_name_val = ['s01_v', 's02_v', 's03_v', 's06_v']
        joints_dir = 'joints3d_25'
        videos_dir = 'videos'
        images_dir = 'images'
        camera_param = 'camera_parameters'
        camera_ids = ['50591643', '58860488', '60457274', '65906101']
    elif args.dataset == "mpii":
        dataset_name = "mpi_inf_3dhp"
        dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
        subset_type = ['train', 'test']
        subj_name_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        subj_name_val = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
        joints_dir = 'joints3d_25'
        videos_dir = 'videos'
        images_dir = 'images'
        camera_param = 'camera_parameters'
        camera_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

    #generate the validation set 
    if args.gen and dataset_name == "humansc3d":
        generate_validation_set(dataset_root_dir, subset_type, subj_name_train)
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

            #data = load_db(base_path, subj_video, joints_dir, images_dir, cams)
            data = load_db_mpii(base_path, subj_video, cams)
            db.extend(data)
            video_count += 1
        dbs.append(db)
        idx_subset_type += 1

    datasets = {'train': dbs[0], 'validation': dbs[1]}

    if args.train:
        with open(os.path.join(dataset_root_dir,'humansc3d_train.pkl'), 'wb') as f:
            pickle.dump(datasets['train'], f)
    
    if args.validation:
        with open(os.path.join(dataset_root_dir, 'humansc3d_test.pkl'), 'wb') as f:
            pickle.dump(datasets['validation'], f)