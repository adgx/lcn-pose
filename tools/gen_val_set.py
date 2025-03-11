import argparse
import os
import os.path as osp

def generate_validation_set(dataset_root_dir, subset_type, subj_name_train):
    print("find training set")
    scs = 'self_contact_signature.json'

    for subj in subj_name_train:
        root_dir = os.path.join(dataset_root_dir, subset_type[0], subj)
        root_dir_val = os.path.join(dataset_root_dir, subset_type[1], subj+'_v')
        #use os.rename(src, dest) to move files

        if os.path.exists(os.path.join(root_dir, scs)):
            if not os.path.exists(root_dir_val):
                    os.makedirs(root_dir_val)
            os.rename(os.path.join(root_dir, scs), os.path.join(root_dir_val, scs))
            print(f"Copy file {scs} is done")
        else:    
            print(f"File {scs} doesn\'t exitst")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if os.path.exists():
                if '':
                    os.rename(src, dest)
            else:    
                print("File exists.")
            print(f"Directory: {dirpath}")
            for dirname in dirnames:
                print(f"  Subdirectory: {os.path.join(dirpath, dirname)}")
            for filename in filenames:
                print(f"  File: {os.path.join(dirpath, filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate validation set from training set')
    args = parser.parse_args()

    dataset_name = 'humansc3d'
    dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
    subset_type = ['train', 'test']
    subj_name_train = ['s01', 's02', 's03', 's06']

    generate_validation_set(dataset_root_dir, subset_type, subj_name_train)