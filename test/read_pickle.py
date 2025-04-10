import numpy as np
import json
import pickle

def main():

    pickle_path = "/Users/andreaongaro/Documents/Documenti Andrea Ongaro/Magistrale/Torino/Corsi/2_ANNO/ComputerVision/Project/lcn-pose/experiment/test3/best_frame_predicted__joint_2025-04-08-17-46.pkl"

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        print(len(data))

        #Print the keys of the dictionary
        for key, value in data.items():
            print(key)
            #Print the shape of the value
            if isinstance(value, np.ndarray):
                print(value.shape)
                print("-------")
            else:
                print(type(value))
                print(value)
                print("-------")

        print(data)

if __name__ == '__main__':
    main()
