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

    result = np.concatenate((data, flipped_data), axis=0)

    return result


def unflip_data(data):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape((2, -1, 17, 3))
    data[1, :, :, 0] *= -1  # flip x of all joints
    data[1, :, left_joints + right_joints] = data[1, :, right_joints + left_joints]
    data = np.mean(data, axis=0)
    data = data.reshape((-1, 17 * 3))

    return data


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
                trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [
                    1,
                    res_h / res_w,
                ]
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
        trainset, testset = trainset.reshape((len(trainset), -1)), testset.reshape(
            (len(testset), -1)
        )

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
                train_labels[idx, :, :2] = item["joint_3d_image"][:, :2] / res_w * 2 - [
                    1,
                    res_h / res_w,
                ]
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
                test_labels[idx, :, :2] = item["joint_3d_image"][:, :2] / res_w * 2 - [
                    1,
                    res_h / res_w,
                ]
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
