import numpy as np
import json
import os

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]

def load_data(data_path):
    """
    Load data from a JSON file.
    Args:
        data_path (str): Path to the JSON file.
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

def get_pred_info(data_pred):
    """
    Get the prediction types from the data.
    Args:
        data_pred (dict): Dictionary of predictions.
    Returns:
        pred_types (list): List of prediction types.
    """
    # this is a big hack
    pred_types = []
    for subj_name in data_pred:
        for action_name in data_pred[subj_name]:
            pred = data_pred[subj_name][action_name]['persons'][0]
            for key in pred:
                if key in ['gpp', 'smplx', 'joints3d', 'blazeposeghum_33']:
                     pred_types.append(key)
            break
        break
    print('Detected prediction types are: ', pred_types)
    return pred_types



def validate_pred_format(data_pred_path, data_template_path):
    try:
        import ipdb; ipdb.set_trace() # debugging starts here
        MAX_FILE_SIZE_MB = 100

        if os.path.getsize(data_pred_path) / (1024*1024.0) > MAX_FILE_SIZE_MB:
            return False, 'File size too large!'

        data_pred = load_data(data_pred_path)
        data_template = load_data(data_template_path)
        pred_types = get_pred_info(data_pred)

        if len(pred_types) == 0:
            return False, 'There should be at least one detected prediction type!'

        if data_pred.keys() != data_template.keys():
            return False, 'Prediction subjects are not the same as the ground truth!'

        for subj_name in data_pred:
            if data_pred[subj_name].keys() != data_template[subj_name].keys():
                return False, 'Actions for subject %s are not the same as in ground truth' % subj_name
            for action_name in data_pred[subj_name]:
                if data_template[subj_name][action_name]['other']['video_fr_ids'] != data_pred[subj_name][action_name]['other']['video_fr_ids']:
                    return False, 'Frames in video_fr_ids are not exactly the same as in the template file!'
                persons_pred = data_pred[subj_name][action_name]['persons']
                persons_template = data_template[subj_name][action_name]['persons']
                if len(persons_pred) != len(persons_template):
                    return False, 'There should be %d persons predicted. Currently there are %d.' % (len(persons_template), len(persons_pred))
                for person_id in range(len(persons_template)):
                    pred = persons_pred[person_id]
                    template = persons_template[person_id]
                    for pred_type in pred_types:
                        if pred_type not in pred:
                            return False, 'Prediction type %s cannot be detected in each sequence!' % pred_type
                        if pred[pred_type].keys() != template[pred_type].keys():
                            return False, 'Prediction type %s in the wrong format!' % pred_type
                        for pred_subtype in pred[pred_type]:
                            if pred_subtype == 'joints3d':
                                for dim in [0, 2]:
                                    if pred[pred_type][pred_subtype].shape[dim] != template[pred_type][pred_subtype].shape[dim]:
                                        return False, 'Shape mismatch %s!' % pred_subtype
                                if pred[pred_type][pred_subtype].shape[1] not in [17, 25]:
                                    return False, 'Shape mismatch %s! Second dimension should be either 17 or 25!'
                            elif pred_subtype == 'blazeposeghum_33':
                                for dim in [0, 2]:
                                    if pred[pred_type][pred_subtype].shape[dim] != template[pred_type][pred_subtype].shape[dim]:
                                        return False, 'Shape mismatch %s!' % pred_subtype
                                if pred[pred_type][pred_subtype].shape[1] not in [33]:
                                    return False, 'Shape mismatch %s! Second dimension should be 33!'
                            else:
                                if pred[pred_type][pred_subtype].shape != template[pred_type][pred_subtype].shape:
                                    return False, 'Shape mismatch %s!' % pred_subtype

        return True, 'Prediction file is valid.'
        
    except Exception as e:
        print(e)
        return False, 'Error! Please validate your prediction file before submitting!'


