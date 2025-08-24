import os
import os.path as osp
import scipy.io as sio
import json 
import numpy as np
import parse
import argparse
import pickle
import re
import math
import mat73
import sys

from data_preparation import convert_humansc3d_mp4_to_image
from gen_val_test_set import generate_val_and_test_set
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
        #get the camera file name
        path_camera = os.path.join(path_cameras, camera_view)
        file_name = os.listdir(path_camera)[0]
        path_camera_file = os.path.join(path_camera, file_name)
        
        with open(path_camera_file, 'r') as cam_json:
            cams_data[camera_view] = json.load(cam_json)
    return cams_data




#Example of  mpii test set camera calibration file
#sensorSize      10 10       # mm
#focalLength     7.32506     # mm
#centerOffset    -0.0322884 0.0929296     # mm
#distortion      0.0 0.0 0.0 0.0 0.0
#origin          3427.28 1387.86 309.42
#up              -0.208215 0.976233 0.06014
#right           0.000575281 0.0616098 -0.9981
#
#From this, we can derive:
#Intrinsics
#
#To go from physical units (mm) to pixels, you need the image resolution, which isn't in your snippet. But assuming you have that (say, W x H = 1024 x 768 or similar), we can calculate:
# Focal length in pixels:
#
#fx = focalLength * (image_width / sensor_width)
#fy = focalLength * pixelAspect
#
#Principal point (optical center):
#
#cx = image_width / 2 + centerOffset_x * (image_width / sensor_width)
#cy = image_height / 2 + centerOffset_y * (image_height / sensor_height)
#
#Distortion:
#
#From the line:
#
#distortion      0.0 0.0 0.0 0.0 0.0
#
#This maps to OpenCV's distortion model:
#
#    k1, k2, k3, p1, p2
#
#Extrinsics
#
#From this block:
#
#origin   3427.28 1387.86 309.42
#up       -0.208215 0.976233 0.06014
#right    0.000575281 0.0616098 -0.9981
#
#You can reconstruct a rotation matrix R from the right, up, and view vectors. The third vector (view) can be calculated as:
#
#view = cross(up, right)
#
#Then, assemble the rotation matrix R using:
#
#R = [right, up, view]^T
#
#The translation vector T is :
#
#T = origin

def load_cams_datatest_mpii(dataset_root_dir, subset):
    dir_util = 'test_util'
    dir_cameras = 'camera_calibration'
    path_cameras = os.path.join(dataset_root_dir, subset, dir_util, dir_cameras)
    #check the path
    if not osp.isdir(path_cameras):
        print(f'Error path isn\'t valid: {path_cameras}')

    cams_data = {}
    image_height = 2048
    image_width = 2048
    calib_files = os.listdir(path_cameras)
    for idx, calib_file in enumerate(calib_files):
        path_camera = os.path.join(path_cameras, calib_file)   
        with open(path_camera, 'r') as cam_cal:
            next(cam_cal)
            
            for line in cam_cal:

                tokens = line.split()
                key, *values = tokens

                if key in ('frame', 'distortionModel','colorCorrection', 'red', 'green', 'blue'):
                    continue
                elif key == 'camera':
                    name = camera_test_ids[idx]  # Store as string, not list
                    cam_data = {}
                elif key in ('sensorSize'):
                    sensor_width = float(values[0])
                    sensor_height = float(values[1])
                elif key in ('focalLength'):
                    focal_length = float(values[0])
                elif key in ('centerOffset'):
                    center_off_x = float(values[0])
                    center_off_y = float(values[1])
                elif key in ('pixelAspect'):
                    pixel_aspect = float(values[0])
                elif key in ('distortion'):
                    k = list(map(float, values[:3]))
                    p = list(map(float, values[3:5]))
                elif key == 'origin':
                    origin = list(map(float, values[:3]))
                elif key == 'up':
                    up = list(map(float, values[:3]))
                elif key == 'right':  # Assuming 'right' marks the end of a camera block
                    right = list(map(float, values[:3]))

                    fx = float(focal_length/(image_width/sensor_width))
                    cam_data['intrinsics_w_distortion'] = {
                        
                        'f': [fx, 
                              float(fx * pixel_aspect)],
                        'c': [float(image_width / 2 + center_off_x * (image_width / sensor_width)),
                              float(image_height / 2 + center_off_y * (image_height / sensor_height))],
                        'k': k,
                        'p': p
                    }
                    cam_data['intrinsics_wo_distortion'] = {
                        'c' : cam_data['intrinsics_w_distortion']['c'],
                        'f' : cam_data['intrinsics_w_distortion']['f']
                    }
                    cam_data['extrinsics'] = {
                        'T' : origin,
                        'R' : [right, up, [a * b for a, b in zip(right, up)]]
                    }
                    cams_data[name] = cam_data
                    break

    return cams_data

#note: for mpii trainset
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
                cam_data['extrinsics'] = {
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

#the action is the ID of the selfcontact action that is show in the video for the dataset humansc3d
def infer_meta_from_name(subj_video, action, cam_id):
    dict_subact_act = {
    i: (0 if i < 117 else 1 if i < 136 else 2)
    for i in range(1, 173)
    }
    dict_subact_act[136] = 0

    meta = {
        'subject': int(subj_video[1:3]),
        'action': int(dict_subact_act[int(action[:3])]),
        'subaction': int(action[:3]),
        'camera': int(cam_id)
    }
    return meta


#A raw way to get information from dirs
def infer_meta_from_name_mpii(subj_video, action, cam_id):
    digit_subj_video = re.search(r'\d+', subj_video).group()
    digit_action = re.search(r'\d+', action).group()


    meta = {
        'subject': int(digit_subj_video),
        'action': int(digit_action),
        'subaction': int(0),
        'camera': cam_id
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
    camera['name'] = str(cameraidx)
    return camera

def _retrieve_camera_mpii(cams, subject, cameraidx):
    R, T = cams[str(cameraidx)]['extrinsics'].values()
    f, c, k, p = cams[str(cameraidx)]['intrinsics_w_distortion'].values() #what about intrinsics_wo_distortion?
    camera = {}
    camera['R'] = R
    camera['T'] = T
    camera['fx'] = f[0]
    camera['fy'] = f[1]
    camera['cx'] = c[0]
    camera['cy'] = c[1]
    camera['k'] = k
    camera['p'] = p
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

def _infer_box_mpii(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 2500.0
    br_joint = root_joint.copy()
    br_joint[:2] += 2500.0
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

def camera_to_image_frame_mpii(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 5000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame

def load_dataitem_mpii(dset, seq_video_anno, cams, camera_id, numimgs, joints_3d_cam, rootIdx = 0):
    meta = infer_meta_from_name_mpii(dset, seq_video_anno, camera_id)
    cam = _retrieve_camera_mpii(cams, meta['subject'], meta['camera'])#handle multicamera pov

    dataset = []

    for i in range(numimgs):
        image = os.path.join(dataset_root_dir, str(camera_id), str(meta['action']), 'frame_'+str(i).zfill(4)+'.jpeg')
        joint_3d_cam = joints_3d_cam[i, :17, :]#obtain the all joints position for the frame
        box = _infer_box_mpii(joint_3d_cam, cam, rootIdx)#obtain info about bounding box
        joint_3d_image = camera_to_image_frame_mpii(joint_3d_cam, box, cam, rootIdx)
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) 
        scale = ((box[2] - box[0]) / 500.0, (box[3] - box[1]) / 500.0)
        dataitem = {
            'videoid': dset+'_'+seq_video_anno,
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
   
    seq_video_dirs = os.listdir(os.path.join(dataset_root_dir, dset))
    
    trainingset = []
    testset = []
    validationset = []
    
    for seq_video_anno in seq_video_dirs:
        print(f'load {dset} {seq_video_anno}')
        annofile = os.path.join(dataset_root_dir, dset, seq_video_anno, 'annot.mat')
        
        if not os.path.exists(annofile):
            print(f"{annofile}: file does not exist")
            
        anno = sio.loadmat(annofile)
        joints_cams = anno['annot3']
        #anno = mat73.loadmat(annofile)
        for idx, joints_3d_cam in enumerate(joints_cams):
            joints_3d_cam = joints_3d_cam[0] 
            joints_3d_cam = np.reshape(joints_3d_cam, (joints_3d_cam.shape[0], 28, 3))
            joints_3d_cam = joints_3d_cam[:, [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16], :]
            numimgs = joints_3d_cam.shape[0]
            print('Formatting done')
        
        

            #used for the traing set
            if args.train:
                if idx <= 7: 
                    trainingset.extend(load_dataitem_mpii(dset, seq_video_anno, cams, camera_ids[idx], numimgs, joints_3d_cam))
            #used for the validation set
            if args.validation is True:
                if idx > 7 and idx < 11:
                    validationset.extend(load_dataitem_mpii(dset, seq_video_anno, cams, camera_ids[idx], numimgs, joints_3d_cam))
            if args.test is True:
                if idx >= 11 and idx <= 13:
                    testset.extend(load_dataitem_mpii(dset, seq_video_anno, cams, camera_ids[idx], numimgs, joints_3d_cam))

    return trainingset, validationset, testset





def load_db_test_mpii(dataset_root_dir, dset, cams, images_dir, rootIdx=0):

    dataset = []
    
    print(f'load {dataset_root_dir}\\test\{dset}')
    annofile = os.path.join(dataset_root_dir, 'test', dset, 'annot_data.mat')
    
    if not os.path.exists(annofile):
        print(f"{annofile}: file does not exist")
        
    anno = mat73.loadmat(annofile)
    valid_frame = anno['valid_frame']
    array_joint = anno['univ_annot3']
    array_joint = np.squeeze(array_joint, axis=2)

    #obtain the vailid index frame
    valid_frame_idx = []
    count = 0 
    for idx, valid in enumerate(valid_frame):
        if valid:
            valid_frame_idx.append(idx)
            #debug count += 1

    array_joint_val = array_joint[:,:,valid_frame_idx] 
    
    #edit array of joints positions
    print(f'Formatting joints data ')
    if int(dset[2:]) < 5:
        cam_name = camera_test_ids[0]
    else: 
        cam_name = camera_test_ids[1]
    
    #get the Traslation vector and rotation matrix of the cam 0
    joints_3d_cam = []
    R_t = np.array(cams[cam_name]['extrinsics']['R']).transpose()
    T = np.array(cams[cam_name]['extrinsics']['T'])
    T = T.reshape(-1, 1)
    T_matrix = np.tile(T, (1, 17))
    num_frames = array_joint_val.shape[2]
    for frame in range(num_frames):
        frame_joints_pos = array_joint_val[:, :, frame]
        joints_pos_w = np.matmul(R_t, (frame_joints_pos - T_matrix)).transpose()
        joints_3d_cam.append(joints_pos_w)
    joints_3d_cam = np.array(joints_3d_cam)
    print('Formatting done')
    
    meta = infer_meta_from_name_mpii(dset, '0', cam_name)
    cam = _retrieve_camera_mpii(cams, meta['subject'], meta['camera'])#handle multicamera pov

    for i in range(num_frames):
        image = os.path.join(dataset_root_dir, dset, images_dir, 'img_'+str(i).zfill(6)+'.jpeg')
        joint_3d_cam = joints_3d_cam[i, :17, :]#obtain the all joints position for the frame
        box = _infer_box_mpii(joint_3d_cam, cam, rootIdx)#obtain info about bounding box
        joint_3d_image = camera_to_image_frame_mpii(joint_3d_cam, box, cam, rootIdx)
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) 
        scale = ((box[2] - box[0]) / 500.0, (box[3] - box[1]) / 500.0)
        dataitem = {
            'videoid': '0',
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
def load_db_humansc3d(dataset_root_dir, dset, cams, joints_dir, images_dir,  rootIdx=0):
    
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
    parser.add_argument("--test", action="store_true", help="Create the pkl for the test set")
    parser.add_argument("-m", "--image", action="store_true", help="Convert dataset mp4 videos to images")
    parser.add_argument("-g", "--gen", action="store_true", help="Generate validation set from training set")
    parser.add_argument("-d", '--dataset', type=str, default="humansc3d", help='Filename of the dataset', choices=["mpii", "humansc3d"])
    parser.add_argument("-c", "--compose", action="store_true", help="Generate a train set that is coposed by humansc3d, mpii and h3.6m")
    args = parser.parse_args()
    
    #debug stuff 
    current_dir = os.getcwd()
    print("Current dir:", current_dir)

    #composing datasets
    if args.compose:
        print("composing datasets: h3.6m, humansc3d, mpii")

        #check that exists the pkl for all dataset
        datasets_dir = os.path.join('..', 'datasets')
        if not os.path.exists(datasets_dir):
            print(f"{datasets_dir}: doesn't exist")
            exit()

        #h3.6m
        dataset_h36m_path = os.path.join(datasets_dir, 'h3.6m', 'h36m_train.pkl') 
        
        if not os.path.exists(dataset_h36m_path):
            print(f"{dataset_h36m_path}: not exist")
            exit()
        
        #humansc3d
        dataset_humansc3d_path = os.path.join(datasets_dir, 'humansc3d', 'humansc3d_train.pkl')
        if not os.path.exists(dataset_humansc3d_path):
            print(f"{dataset_humansc3d_path}: not exist")
            exit()

        #mpii
        dataset_mpii_path = os.path.join(datasets_dir, 'mpi_inf_3dhp', 'mpii_train.pkl')
        if not os.path.exists(dataset_mpii_path):
            print(f"{dataset_mpii_path}: not exist")

        #open h3.6m
        with open(dataset_h36m_path, 'rb') as h36m_pkl:
            dataset_h36m = pickle.load(h36m_pkl)

        #open humansc3d
        with open(dataset_humansc3d_path, 'rb') as humansc3d_pkl:
            dataset_humansc3d = pickle.load(humansc3d_pkl)

        #open mpii
        with open(dataset_mpii_path, 'rb') as mpii_pkl:
            dataset_mpii = pickle.load(mpii_pkl)
        #determinate the min number of samples
        num_mpii_samples = len(dataset_mpii)
        num_humansc3d_samples = len(dataset_humansc3d)
        num_h36m_samples = len(dataset_h36m)
        min_samples = min([num_h36m_samples, num_humansc3d_samples, num_mpii_samples])
        
        percentage = 0.33

        num_sample = math.ceil(percentage*min_samples)

        dataset_composed = []
        dataset_composed.extend(dataset_h36m[:num_sample])
        dataset_composed.extend(dataset_humansc3d[:num_sample])
        dataset_composed.extend(dataset_mpii[:num_sample])

        dataset_composed_path = os.path.join(datasets_dir, 'dataset_composed')
        if not os.path.exists(dataset_composed_path):
            os.makedirs(dataset_composed_path)
        with open(os.path.join(dataset_composed_path, f'db_composed.pkl'), 'wb') as f:
                pickle.dump(dataset_composed, f)

        exit()
        
        #ex: 4subject * 100000 *0.33 = 132000

    load_db = None
    #information to retrieve the dataset information

    if args.dataset == "humansc3d":
        dataset_name = args.dataset
        dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
        subset_type = ['train', 'val', 'test']
        subj_name_train = ['s01', 's02', 's03', 's06']
        subj_name_val = ['s01_v', 's02_v', 's03_v', 's06_v']
        subj_name_test = ['s01_t', 's02_t', 's03_t', 's06_t']
        joints_dir = 'joints3d_25'
        videos_dir = 'videos'
        images_dir = 'images'
        camera_param = 'camera_parameters'
        camera_ids = ['50591643', '58860488', '60457274', '65906101']
        load_db = load_db_humansc3d
    elif args.dataset == "mpii":
        dataset_name = "mpi_inf_3dhp"
        dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
        subset_type = ['train', 'test']
        subj_name_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        subj_name_val = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
        joints_dir = 'joints3d_25'
        videos_dir = 'videos'
        images_dir = 'images'
        images_test_dir = 'imageSequence'
        camera_param = 'camera_parameters'
        camera_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        camera_test_ids = ['cam_0', 'cam_8']
        load_db = load_db_mpii

    if load_db is None:
        print("Error loader for the dataset not found")
        exit(-1)

    
    #generate the validation set 
    if args.gen and dataset_name == "humansc3d":
        generate_val_and_test_set(dataset_root_dir, subset_type, subj_name_train)
    # loading cameras data
    # we assuming that camera parameters doesn't change in the time and keep the same between train and validation set
    cams = load_cams_data(dataset_root_dir, subset_type[0], subj_name_train[0], camera_param)
    
    if args.dataset == 'mpii':

        mpii_dbs = [[], [], []]

        base_dir = osp.join(dataset_root_dir, subset_type[0])
        
        for subj in subj_name_train:
            #check if the subject dir exist
            if not osp.isdir(osp.join(base_dir, subj)):
                print(f'subject: {subj} not found')
                continue
                
            #loading data from training set and pushes the last three point view into the validation set  
            trainingset, validationset, testset = load_db_mpii(base_dir, subj, cams, rootIdx=0)
            mpii_dbs[0].extend(trainingset) 
            mpii_dbs[1].extend(validationset)
            mpii_dbs[2].extend(testset)
#
        ##loading the mpii validation set
        #if args.validation:
        #    cams_test = load_cams_datatest_mpii(dataset_root_dir, subset_type[1])
        #    
        #    for subj in subj_name_val:
        #        if not osp.isdir(osp.join(dataset_root_dir, subset_type[1],subj)):
        #            print(f'subject: {subj} not found')
        #            continue
        #        data = load_db_test_mpii(dataset_root_dir, subj, cams_test, images_test_dir, rootIdx=0)
        #        mpii_dbs[1].extend(data)
        #    
        #generate the traing set
        if args.train:
            with open(os.path.join(dataset_root_dir,f'{args.dataset}_train.pkl'), 'wb') as f:
                pickle.dump(mpii_dbs[0], f)
        #generate the validation set
        if args.validation:
            with open(os.path.join(dataset_root_dir, f'{args.dataset}_val.pkl'), 'wb') as f:
                pickle.dump(mpii_dbs[1], f)
        #generate the test set
        if args.test:
            with open(os.path.join(dataset_root_dir, f'{args.dataset}_test.pkl'), 'wb') as f:
                pickle.dump(mpii_dbs[2], f)
        exit()
    
    train_dirs = []
    val_dirs = []
    test_dirs = []
    #allow to find the train and validation set
    if args.train:
        train_dirs = find_dirs(dataset_root_dir, subset_type[0], subj_name_train)

    if args.validation:
        val_dirs = find_dirs(dataset_root_dir, subset_type[1], subj_name_val)

    if args.test:
        test_dirs = find_dirs(dataset_root_dir, subset_type[2], subj_name_test)

    train_val_test_datasets = [train_dirs, val_dirs, test_dirs]
    dbs = []
    video_count = 0
    idx_subset_type = 0 

    if dataset_name == 'humansc3d':

        for dataset in train_val_test_datasets:
            db = []
            for subj_video in dataset:
                base_path = os.path.join(dataset_root_dir, subset_type[idx_subset_type], subj_video)
                if np.mod(video_count, 1) == 0:
                    print('Process {}: {}'.format(video_count, subj_video))
    
                if args.image and dataset_name == 'humansc3d':  
                    convert_humansc3d_mp4_to_image(base_path,  videos_dir, images_dir, camera_ids)
    
                data = load_db(base_path, subj_video, cams, joints_dir, images_dir)
                db.extend(data)
                video_count += 1
            dbs.append(db)
            idx_subset_type += 1
    
        datasets = {'train': dbs[0], 'val': dbs[1], 'test': dbs[2]}
    
        if args.train:
            with open(os.path.join(dataset_root_dir,f'{args.dataset}_train.pkl'), 'wb') as f:
                pickle.dump(datasets['train'], f)
        
        if args.validation:
            with open(os.path.join(dataset_root_dir,f'{args.dataset}_val.pkl'), 'wb') as f:
                pickle.dump(datasets['val'], f)
        if args.test:
            with open(os.path.join(dataset_root_dir,f'{args.dataset}_test.pkl'), 'wb') as f:
                pickle.dump(datasets['test'], f)