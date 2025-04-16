import numpy as np

def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: Data read from the JSON file.
    """
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data)