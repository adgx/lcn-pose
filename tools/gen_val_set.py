import argparse
import os
import os.path as osp
import math
import shutil

def generate_validation_set(dataset_root_dir, subset_type, subj_name_train, percent=0.20):
    print("find training set")
    scs = 'self_contact_signature.json'

    for subj in subj_name_train:
        root_dir = os.path.join(dataset_root_dir, subset_type[0], subj)
        root_dir_val = os.path.join(dataset_root_dir, subset_type[1], subj+'_v')
        #use os.rename(src, dest) to move files
        #copy the self_contact_signature.json for each subj
        if os.path.exists(os.path.join(root_dir, scs)):
            if not os.path.exists(root_dir_val):
                    os.makedirs(root_dir_val)
            if not os.path.exists(os.path.join(root_dir_val, scs)):
                shutil.copy(os.path.join(root_dir, scs), os.path.join(root_dir_val, scs))
            print(f"Copy file {scs} is done")
        else:    
            print(f"File {scs} doesn\'t exitst")
        
        #to go through directory tree 
        for dirpath, dirnames, filenames in os.walk(root_dir):
            num_files = len(filenames)
            #handle the move of the frame videos
            if os.path.basename(os.path.dirname(dirpath)) == 'images':
                print("Move frames")
                dirnames.sort()
                num_dirs = len(dirnames)
                num_files_move = math.floor(num_dirs * percent)
                dst = dirpath.replace(f'{subj}', subj+'_v').replace(f'{subset_type[0]}', f'{subset_type[1]}')
                print(f'Move to: {dst} from: {dirpath}')
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for dirframe in dirnames[-num_files_move:]:
                    print(f'Move dir: {file}')
                    if not os.path.exists(os.path.join(dst, dirframe)):
                        os.rename(os.path.join(dirpath,dirframe), os.path.join(dst, dirframe))
            
            #handle the remaining samples files of the trian set
            if (num_files > 1) and not 'images' in os.listdir(dirpath):
                print('Move files')
                filenames.sort()
                num_files_move = math.floor(num_files * percent)
                dst = dirpath.replace(f'{subj}', subj+'_v').replace(f'{subset_type[0]}', f'{subset_type[1]}')
                print(f'Move to: {dst} from: {dirpath}')
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for file in filenames[-num_files_move:]:
                    print(f'Move file: {file}')
                    if not os.path.exists(os.path.join(dst, file)):
                        os.rename(os.path.join(dirpath, file), os.path.join(dst, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate validation set from training set')
    args = parser.parse_args()

    dataset_name = 'humansc3d'
    dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
    subset_type = ['train', 'test']
    subj_name_train = ['s01']

    generate_validation_set(dataset_root_dir, subset_type, subj_name_train)