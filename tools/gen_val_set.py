import argparse
import os
import os.path as osp
import math
import shutil
import numpy as np

def generate_validation_set(dataset_root_dir, subset_type, subj_name_train, percent=0.20):
    
    #indices of the video where are done these action
    num_annot= 172 #humansc3d has 172 total annotation
    standing_act_idxs = list(range(116)) + [135]
    sitting_act_idxs = list(range(117, 136))
    interact_chair_act_idxs = list(range(137, 172))
    #percentage of action in the trainset
    percent_standing = 0.68
    percent_sitting = 0.11
    percent_interact_chair = 0.21
    num_files_move = math.floor(num_annot * percent)
    num_standing_act = math.floor(num_files_move * percent_standing) #23
    num_sitting_act = math.floor(num_files_move * percent_sitting) + 1 #4
    num_interact_act = math.floor(num_files_move * percent_interact_chair) #7

    print("find training set")
    scs = 'self_contact_signature.json'

    for subj in subj_name_train:
        idxs = np.random.permutation(standing_act_idxs)[:num_standing_act]
        idxs = np.append(idxs, np.random.permutation(sitting_act_idxs)[:num_sitting_act])
        idxs = np.append(idxs, np.random.permutation(interact_chair_act_idxs)[:num_interact_act])
        root_dir = os.path.join(dataset_root_dir, subset_type[0], subj)# subject train dir
        root_dir_val = os.path.join(dataset_root_dir, subset_type[1], subj+'_v')#dir for the valuation set
        #use os.rename(src, dest) to move files
        #copy the self_contact_signature.json for each subj
        if os.path.exists(os.path.join(root_dir, scs)):
            if not os.path.exists(root_dir_val):
                    os.makedirs(root_dir_val)#make valuation dir
            if not os.path.exists(os.path.join(root_dir_val, scs)):
                shutil.copy(os.path.join(root_dir, scs), os.path.join(root_dir_val, scs))#copy the self_contact_signature.json
            print(f"Copy file {scs} is done")
        else:    
            print(f"File {scs} doesn\'t exitst")
        
        #to go through directory tree 
        for dirpath, dirnames, filenames in os.walk(root_dir):
            num_files = len(filenames)
            #handle the move of the frame videos
            #if os.path.basename(os.path.dirname(dirpath)) == 'images':
            #    print("Move frames")
            #    dirnames.sort()
            #    num_dirs = len(dirnames)
            #    num_files_move = math.floor(num_dirs * percent)
            #    dst = dirpath.replace(f'{subj}', subj+'_v').replace(f'{subset_type[0]}', f'{subset_type[1]}')
            #    print(f'Move to: {dst} from: {dirpath}')
            #    if not os.path.exists(dst):
            #        os.makedirs(dst)
            #    for dirframe in dirnames[-num_files_move:]:
            #        print(f'Move dir: {file}')
            #        if not os.path.exists(os.path.join(dst, dirframe)):
            #            os.rename(os.path.join(dirpath,dirframe), os.path.join(dst, dirframe))
            
            #handle the remaining samples files of the trian set
            if (num_files > 1) and not 'images' in os.listdir(dirpath):
                print('Move files')
                filenames.sort()
                
                filenames_extract = [filenames[i] for i in idxs]

                dst = dirpath.replace(f'{subj}', subj+'_v').replace(f'{subset_type[0]}', f'{subset_type[1]}')
                print(f'Move to: {dst} from: {dirpath}')
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for file in filenames_extract:
                    print(f'Move file: {file}')
                    if not os.path.exists(os.path.join(dst, file)):
                        shutil.move(os.path.join(dirpath, file), os.path.join(dst, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate validation set from training set')
    args = parser.parse_args()

    dataset_name = 'humansc3d'
    dataset_root_dir = os.path.join( '..', 'datasets', dataset_name)
    subset_type = ['train', 'test']
    subj_name_train = ['s01', 's02', 's03', 's06']

    generate_validation_set(dataset_root_dir, subset_type, subj_name_train)