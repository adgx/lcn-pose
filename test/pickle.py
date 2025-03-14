import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def main():
    import pickle

    pickle_path = "/Users/andreaongaro/Documents/Documenti Andrea Ongaro/Magistrale/Torino/Corsi/2_ANNO/ComputerVision/Project/lcn-pose/dataset/h36m_test.pkl"

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        print(len(data))
        data = data[0]

        for keys, values in data.items():
            print(keys)
            print(values)
            print("\n")

    json_path = "/Users/andreaongaro/Documents/Documenti Andrea Ongaro/Magistrale/Torino/Corsi/2_ANNO/ComputerVision/Project/lcn-pose/dataset/h36m_test.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)



if __name__ == '__main__':
    main()
