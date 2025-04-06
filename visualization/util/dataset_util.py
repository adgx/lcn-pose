import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    

#visualize_prediction
def read_video(vid_path):
    """
    Read video and return frames as numpy array.
    Args:
        vid_path (str): Path to the video file.
    Returns:
        frames (numpy array): Array of frames.
    """
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
    cap.release()
    frames = np.array(frames)
    return frames

#visualize_prediction
def read_cam_params(cam_path):
    """
    Read camera parameters from json file.
    Args:
        cam_path (str): Path to the camera parameters json file.
    Returns:
        cam_params (dict): Dictionary of camera parameters.
    """
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
    return cam_params

#visualize_prediction & visualization_lab_dataset
def project_3d_to_2d(points3d, intrinsics, intrinsics_type):
    """
    Project 3D points to 2D using camera intrinsics.
    Args:
        points3d (numpy array): 3D points to project.
        intrinsics (dict): Camera intrinsics.
        intrinsics_type (str): Type of camera intrinsics ('w_distortion' or 'wo_distortion').
    Returns:
        proj (numpy array): Projected 2D points.
    """
    p = intrinsics['p'][:, [1, 0]] #inverte
    x = points3d[:, :2] / points3d[:, 2:3]
    r2 = np.sum(x**2, axis=1)
    radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
    tan = np.matmul(x, np.transpose(p))
    xx = x*(tan + radial) + r2[:, np.newaxis] * p
    return intrinsics['f'] * xx + intrinsics['c']

#visualize_prediction & visualization_lab_dataset
def plot_over_image(frame, points_2d=np.array([]), with_ids=True, with_limbs=True, path_to_write=None):
    """
    Plot 2D points over the image.
    
    Args:
        frame (numpy array): Image frame.
        points_2d (numpy array): 2D points to plot.
        with_ids (bool): Whether to show point IDs.
        with_limbs (bool): Whether to show limbs.
        path_to_write (str): Path to save the image.
    
    Returns:
        None
    """
    num_points = points_2d.shape[0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    if points_2d.shape[0]:
        #ax.plot(points_2d[:, 0], points_2d[:, 1], 'x', markeredgewidth=10, color='white')
        if with_ids:
            for i in range(num_points):
                ax.text(points_2d[i, 0], points_2d[i, 1], str(i), color='red', fontsize=20)
        if with_limbs:
            limbs = [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                    [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]]
            for limb in limbs:
                if limb[0] < num_points and limb[1] < num_points:
                    ax.plot([points_2d[limb[0], 0], points_2d[limb[1], 0]], 
                            [points_2d[limb[0], 1], points_2d[limb[1], 1]],
                            linewidth=12.0)
            
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches = 0, bbox_inches='tight')


def show_image(frame):
    """
    Show image using matplotlib.
    
    Args:
        frame (numpy array): Image frame.
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    plt.axis('off')
    plt.show()
    plt.close(fig)
#visualize_prediction
def load_data(data_path):
    """
    Load data from json file.
    Args:
        data_path (str): Path to the json file.
    Returns:
        data (dict): Dictionary of data.
    """
    with open(data_path) as f:
        data = json.load(f)
    for subj_name in data:
        for action_name in data[subj_name]:
            for person_id in range(len(data[subj_name][action_name]['persons'])):
                for data_type in data[subj_name][action_name]['persons'][person_id]:
                    for key in data[subj_name][action_name]['persons'][person_id][data_type]:
                        if type(data[subj_name][action_name]['persons'][person_id][data_type][key]) is list:
                            data[subj_name][action_name]['persons'][person_id][data_type][key] = np.array(data[subj_name][action_name]['persons'][person_id][data_type][key])
            if 'cam_params' in data[subj_name][action_name]['other']:
                for key1 in data[subj_name][action_name]['other']['cam_params']:
                    for key2 in data[subj_name][action_name]['other']['cam_params'][key1]:
                        data[subj_name][action_name]['other']['cam_params'][key1][key2] = np.array(data[subj_name][action_name]['other']['cam_params'][key1][key2])
    return data



def load_data_from_pickle(data_path):
    import pickle
    results = []
    with open(data_path, 'rb') as f:
        return pickle.load(f)["result"]

def load_data_from_picklev2(data_path):
    import pickle
    results = []
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def remove_wrong_joints(j2d, frame):
    """
    Remove wrong joints from the dataset.
    Args:
        None
    Returns:
        None
    """

            
    
    return j2d

## Used in visualization_lab_dataset.py
def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, subject="w_markers"):
    """"""
    if subject == 'wo_markers':
        assert(dataset_name == 'chi3d')
    vid_path = '%s/%s/%s/%s/videos/%s/%s.mp4' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    cam_path = '%s/%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
    gpp_path = '%s/%s/%s/%s/gpp/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
    smplx_path = '%s/%s/%s/%s/smplx/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)

    cam_params = read_cam_params(cam_path)

    with open(j3d_path) as f:
        j3ds = np.array(json.load(f)['joints3d_25'])
    seq_len = j3ds.shape[-3]
    with open(gpp_path) as f:
        gpps = json.load(f)
    with open(smplx_path) as f:
        smplx_params = json.load(f)
    frames = read_video(vid_path)[:seq_len]
    
    dataset_to_ann_type = {'chi3d': 'interaction_contact_signature', 
                           'fit3d': 'rep_ann', 
                           'humansc3d': 'self_contact_signature'}
    ann_type = dataset_to_ann_type[dataset_name]
    annotations = None
    if ann_type:
        ann_path = '%s/%s/%s/%s/%s.json' % (data_root, dataset_name, subset, subj_name, ann_type)
        with open(ann_path) as f:
            annotations = json.load(f)
    
    if dataset_name == 'chi3d': # 2 people in each frame
        subj_id = 0 if subject == "w_markers" else 1
        j3ds = j3ds[subj_id, ...]
        for key in gpps:
            gpps[key] = gpps[key][subj_id]
        for key in smplx_params:
            smplx_params[key] = smplx_params[key][subj_id]
        
    
    return frames, j3ds, cam_params, gpps, smplx_params, annotations







