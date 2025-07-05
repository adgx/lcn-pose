import numpy as np
import os, sys
import pickle, h5py

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

#GT_TEST_PATH = os.path.join(ROOT_PATH, "dataset/h36m_test.pkl")
#GT_TRAIN_PATH = os.path.join(ROOT_PATH, "dataset/h36m_train.pkl")


def flip_data(data):
    """
    horizontal flip
        data: [N, 17*k] or [N, 17, k], i.e. [x, y], [x, y, confidence] or [x, y, z]
    Return
        result: [2N, 17*k] or [2N, 17, k]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    flipped_data = data.copy().reshape((len(data), 17, -1))
    flipped_data[:, :, 0] *= -1  # flip x of all joints
    flipped_data[:, left_joints + right_joints] = flipped_data[
        :, right_joints + left_joints
    ]
    flipped_data = flipped_data.reshape(data.shape)

    #result = np.concatenate((data, flipped_data), axis=0)

    return flipped_data

def translation_data(data, translation_factor=2):
    """
    Translate data of a given factor
        data: [N, 17*k] or [N, 17, k] -> k can be 2 or 3 in our case
    Return
        result: [N, 17*k] or [N, 17, k]

    This function adds a random translation to the data.
    The translation is a random integer in the range [-translation_factor, translation_factor].
    The data is reshaped to ensure it has the correct dimensions.
    """
    data_copied = data.copy().reshape((len(data), 17, -1))
    data_copied += translation_factor
    return data_copied.reshape(data.shape) #TODO: fix this to return the correct shape

def get_subset_by_camera(gt_dataset, subset_size=1000):
    """
    Get a subset of the dataset.
        gt_dataset: list of dictionaries containing the ground truth data
        subset_size: number of items to include in the subset
    Return
        subset: list of dictionaries containing the subset of the dataset
    """

    # check if cameraid is present in the dataset
    if not all("cameraid" in item for item in gt_dataset):
        raise ValueError("The dataset must contain 'cameraid' key in each item.")
    
    # Get distinct camera names
    camera_names = set(item["cameraid"] for item in gt_dataset)

    # Divide the dataset into n. subsets based on camera names
    subsets = {camera_name: [] for camera_name in camera_names}
   
    # Randomly select items for each camera to create the subset until the subset size divided by the number of cameras is reached
    for item in gt_dataset:
        camera_name = item["cameraid"]
        if len(subsets[camera_name]) < subset_size // len(camera_names):
            subsets[camera_name].append(item)
        
    # Combine the subsets into a single list
    return [gt_dataset_item for subset in subsets.values() for gt_dataset_item in subset]
    

def get_subset_by_action(gt_dataset, subset_size=1000):
    """
    Get a subset of the dataset.
        gt_dataset: list of dictionaries containing the ground truth data
        subset_size: number of items to include in the subset
    Return
        subset: list of dictionaries containing the subset of the dataset
    """
    # check if action is present in the dataset
    if not all("action" in item for item in gt_dataset):
        raise ValueError("The dataset must contain 'action' key in each item.")
    # Get distinct action names
    action_names = set(item["action"] for item in gt_dataset)

    # Divide the dataset into n. subsets based on action names
    subsets = {action_name: [] for action_name in action_names}
   
    # Randomly select items for each action to create the subset until the subset size divided by the number of actions is reached
    for item in gt_dataset:
        action_name = item["action"]
        if len(subsets[action_name]) < subset_size // len(action_names):
            subsets[action_name].append(item)
        
    # Combine the subsets into a single list
    return [gt_dataset_item for subset in subsets.values() for gt_dataset_item in subset]

def get_subset_by_subject(gt_dataset, subset_size=1000):
    """
    Get a subset of the dataset.
        gt_dataset: list of dictionaries containing the ground truth data
        subset_size: number of items to include in the subset
    Return
        subset: list of dictionaries containing the subset of the dataset
    """
    # check if subject is present in the dataset
    if not all("subject" in item for item in gt_dataset):
        raise ValueError("The dataset must contain 'subject' key in each item.")
    
    # Get distinct subject names
    subject_names = set(item["subject"] for item in gt_dataset)

    # Divide the dataset into n. subsets based on subject names
    subsets = {subject_name: [] for subject_name in subject_names}
   
    # Randomly select items for each subject to create the subset until the subset size divided by the number of subjects is reached
    for item in gt_dataset:
        subject_name = item["subject"]
        if len(subsets[subject_name]) < subset_size // len(subject_names):
            subsets[subject_name].append(item)
        
    # Combine the subsets into a single list
    return [gt_dataset_item for subset in subsets.values() for gt_dataset_item in subset]

def get_subset(gt_dataset, subset_size=1000, mode="camera"):
    """
    Get a subset of the dataset.
        gt_dataset: list of dictionaries containing the ground truth data
        subset_size: number of items to include in the subset
        mode: "camera" or "action" or "camera_action" or "subject"
    Return
        subset: list of dictionaries containing the subset of the dataset
    """
    if subset_size == None:
        return gt_dataset
    if mode == "camera":
        return get_subset_by_camera(gt_dataset, subset_size)
    elif mode == "action":
        return get_subset_by_action(gt_dataset, subset_size)
    elif mode == "subject":
        return get_subset_by_subject(gt_dataset, subset_size)  # Assuming subject is similar to action
    elif mode == "camera_action":
        # Get a subset of the dataset by camera and action
        camera_subset = get_subset_by_camera(gt_dataset, subset_size)
        action_subset = get_subset_by_action(gt_dataset, subset_size)
        
        # Combine the two subsets
        combined_subset = list(set(camera_subset) & set(action_subset))
        
        # If the combined subset is smaller than the requested size, return it
        if len(combined_subset) < subset_size:
            return combined_subset
        
        # Otherwise, randomly select items from the combined subset to reach the requested size
        return np.random.choice(combined_subset, size=subset_size, replace=False).tolist()
    elif mode == "camera_subject":
        # Get a subset of the dataset by camera and subject
        camera_subset = get_subset_by_camera(gt_dataset, subset_size)
        subject_subset = get_subset_by_subject(gt_dataset, subset_size)
        
        # Combine the two subsets
        combined_subset = list(set(camera_subset) & set(subject_subset))
        
        # If the combined subset is smaller than the requested size, return it
        if len(combined_subset) < subset_size:
            return combined_subset
        
        # Otherwise, randomly select items from the combined subset to reach the requested size
        return np.random.choice(combined_subset, size=subset_size, replace=False).tolist()
    

def untranslation_data(data, translation_factor=2, number_actions=2):
    """
    Average original data and translated data
        data: [2N, 17*k] or [2N, 17, k]
    Return
        result: [N, 17*k] or [N, 17, k]
    """
    data = data.copy().reshape( -1, 17, 3)
    data[:, :, 0] *= -1
    
    return data

def unflip_data(data, number_actions=2):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape((number_actions, -1, 17, 3))
    data[1, :, :, 0] *= -1  # flip x of all joints
    data[1, :, left_joints + right_joints] = data[1, :, right_joints + left_joints]
    data = data.reshape((-1, 17 * 3))
    return data

def unflip_data(data):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape(-1, 17, 3)
    data[:, :, 0] *= -1  # flip x of all joints
    data[:, left_joints + right_joints] = data[:, right_joints + left_joints]
    return data

#def undo(data, translation_factor=2, number_actions=2):
#    """
#    Average original data, flipped data, rotated data and translated data
#        data: [2N, 17*3]
#    Return
#        result: [N, 17*3]
#    """
#    # Untranslation
#    #data = untranslation_data(data, translation_factor, number_actions)
#    
#    # Unflip
#    #data = unflip_data(data, number_actions)
#    
#    # Average the results
#    data = np.mean(data, axis=0)  # [N, 17*3]
#    data = data.reshape((-1, 17 * 3))
#    return data

def undo(data, translation_factor=2, number_actions=2):
    """
    Average original data, flipped data, rotated data and translated data
        data: [N*(nop+1), 17*3]
    Return
        result: [N, 17*3]
    """
    # Untranslation
    data = untranslation_data(data, translation_factor, number_actions)
    
    # Unflip
    data = unflip_data(data, number_actions)
    
    # Average the results
    data = np.mean(data.reshape((number_actions, -1, 17 * 3)), axis=0)  # [N, 17*3]
    data = data.reshape((-1, 17 * 3))
    return data

#rotate 
def rotate_data(data, angle = 180):
    """
    rotate points
    """
    #rotation angle in radians
    theta = np.radians(angle)
    #rotation matrix for z-axis
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    rotated_data = data.copy().reshape((len(data), 17, -1))
    
    #rotation
    for idx, jointsPoint in enumerate(rotated_data):
        pivot = jointsPoint[0,:2].copy()
        jointsPoint[:, :2] -= pivot
        
        #case [x, y, confidence], [x, y, z]
        if rotated_data.shape[2] == 3:
            jointsPoint =  jointsPoint @ Rz.T
        #case [x, y]
        else:
            jointsPoint = jointsPoint @ Rz.T[:2, :2]

        jointsPoint[:, :2] += pivot

        rotated_data[idx] = jointsPoint

    rotated_data = rotated_data.reshape(data.shape)
    #result = np.concatenate((data, rotated_data), axis=0)
    
    return rotated_data 

class DataReader(object):
    def __init__(self):
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None

    def real_read(self, subset, type):
        file_name = "%s_%s.pkl" % (subset, type)
        print("loading %s" % file_name)
        file_path = os.path.join(ROOT_PATH, "dataset", file_name)
        with open(file_path, "rb") as f:
            gt = pickle.load(f)
        return gt

    def read_2d(self, gt_trainset, gt_testset, which="scale", read_confidence=True):
        if self.gt_trainset is None:
            self.gt_trainset = gt_trainset
        if self.gt_testset is None:
            self.gt_testset = gt_testset

        trainset = np.empty((len(self.gt_trainset), 17, 2))  # [N, 17, 2]
        testset = np.empty((len(self.gt_testset), 17, 2))  # [N, 17, 2]
        for idx, item in enumerate(self.gt_trainset):
            trainset[idx] = item["joint_3d_image"][:, :2]
        for idx, item in enumerate(self.gt_testset):
            testset[idx] = item["joint_3d_image"][:, :2]
        if read_confidence:
                train_confidence = np.ones((len(self.gt_trainset), 17, 1))  # [N, 17, 1]
                test_confidence = np.ones((len(self.gt_testset), 17, 1))  # [N, 17, 1]

        # normalize
        if which == "scale":
            # map to [-1, 1]
            
            #Trainset
            for idx, item in enumerate(self.gt_trainset):
                camera_name = str(item["camera_param"]["name"])
                if camera_name == "54138969" or camera_name == "60457274":
                    res_w, res_h = 1000, 1002
                elif camera_name == "55011271" or camera_name == "58860488" or camera_name == "50591643" or camera_name == "65906101":
                    res_w, res_h = 1000, 1000
                elif camera_name.find("cam_") != -1:
                    res_w, res_h = 2048, 2048
                elif int(camera_name) >= 0:
                    res_w, res_h = 2048, 2048
                else:
                    assert 0, "%d data item has an invalid camera name" % idx
                trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1,res_h / res_w]
            
            #Testset
            for idx, item in enumerate(self.gt_testset):
                camera_name = str(item["camera_param"]["name"])
                if camera_name == "54138969" or camera_name == "60457274":
                    res_w, res_h = 1000, 1002
                elif camera_name == "55011271" or camera_name == "58860488" or camera_name == "50591643" or camera_name == "65906101":
                    res_w, res_h = 1000, 1000
                elif camera_name.find("cam_") != -1:
                    res_w, res_h = 2048, 2048
                elif int(camera_name) >= 0:
                    res_w, res_h = 2048, 2048
                else:
                    assert 0, "%d data item has an invalid camera name" % idx
                testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        else:
            assert 0, "not support normalize type %s" % which

        if read_confidence:
            trainset = np.concatenate(
                (trainset, train_confidence), axis=2
            )  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]

        # reshape
        trainset, testset = trainset.reshape((len(trainset), -1)), testset.reshape((len(testset), -1))

        return trainset, testset

    def read_3d(self, which="scale"):
        if self.gt_trainset is None:
            self.gt_trainset = self.real_read("train")
        if self.gt_testset is None:
            self.gt_testset = self.real_read("test")

        # normalize
        train_labels = np.empty((len(self.gt_trainset), 17, 3))
        test_labels = np.empty((len(self.gt_testset), 17, 3))
        if which == "scale":
            # map to [-1, 1]
            for idx, item in enumerate(self.gt_trainset):
                camera_name = str(item["camera_param"]["name"])
                if camera_name == "54138969" or camera_name == "60457274":
                    res_w, res_h = 1000, 1002
                elif camera_name == "55011271" or camera_name == "58860488" or camera_name == "50591643" or camera_name == "65906101":
                    res_w, res_h = 1000, 1000
                elif camera_name.find("cam_") != -1:
                    res_w, res_h = 2048, 2048
                elif int(camera_name) >= 0:
                    res_w, res_h = 2048, 2048
                else:
                    assert 0, "%d data item has an invalid camera name" % idx
                train_labels[idx, :, :2] = item["joint_3d_image"][:, :2] / res_w * 2 - [1, res_h / res_w]
                train_labels[idx, :, 2:] = item["joint_3d_image"][:, 2:] / res_w * 2
            for idx, item in enumerate(self.gt_testset):
                camera_name = str(item["camera_param"]["name"])
                if camera_name == "54138969" or camera_name == "60457274":
                    res_w, res_h = 1000, 1002
                elif camera_name == "55011271" or camera_name == "58860488" or camera_name == "50591643" or camera_name == "65906101":
                    res_w, res_h = 1000, 1000
                elif camera_name.find("cam_") != -1:
                    res_w, res_h = 2048, 2048
                elif int(camera_name) >= 0:
                    res_w, res_h = 2048, 2048
                else:
                    assert 0, "%d data item has an invalid camera name" % idx
                test_labels[idx, :, :2] = item["joint_3d_image"][:, :2] / res_w * 2 - [1,res_h / res_w]
                test_labels[idx, :, 2:] = item["joint_3d_image"][:, 2:] / res_w * 2
        else:
            assert 0, "not support normalize type %s" % which

        # reshape
        train_labels, test_labels = train_labels.reshape(
            (-1, 17 * 3)
        ), test_labels.reshape((-1, 17 * 3))

        return train_labels, test_labels

    def denormalize(self, data, which="scale"):
        if self.gt_testset is None:
            self.gt_testset = self.real_read("test")

        if which == "scale":
            data = data.reshape((-1, 17, 3)).copy()
            # denormalize (x,y,z) coordiantes for results
            for idx, item in enumerate(self.gt_testset):
                camera_name = str(item["camera_param"]["name"])
                if camera_name == "54138969" or camera_name == "60457274":
                    res_w, res_h = 1000, 1002
                elif camera_name == "55011271" or camera_name == "58860488"  or camera_name == "50591643" or camera_name == "65906101":
                    res_w, res_h = 1000, 1000
                elif camera_name.find("cam_") != -1:
                    res_w, res_h = 2048, 2048
                elif int(camera_name) >= 0:
                    res_w, res_h = 2048, 2048
                else:
                    assert 0, "%d data item has an invalid camera name" % idx

                data[idx, :, :2] = (data[idx, :, :2] + [1, res_h / res_w]) * res_w / 2
                data[idx, :, 2:] = data[idx, :, 2:] * res_w / 2
        else:
            assert 0
        return data


if __name__ == "__main__":
    datareader = DataReader()
    train_data, test_data, train_2d_mean, train_2d_std = datareader.read_2d(
        which="scale", mode="dt_ft", read_confidence=False
    )
    train_labels, test_labels, train_3d_mean, train_3d_std = datareader.read_3d(
        which="scale", mode="dt_ft"
    )
