import pickle
import tensorflow as tf
import numpy as np
from tools import tools, params_help, data
from network import  models_att # models_attr2 as
import os
import argparse
import pprint
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



def parse_args():
    parser = argparse.ArgumentParser(description='train')

    # optional arguments
    parser.add_argument('--test-indices', help='test idx ', type=str)
    parser.add_argument('--mask-type', help='mask type ', type=str)
    parser.add_argument('--graph', help='index of graphs', type=int, default=0)
    parser.add_argument('--knn', help='expand of neighbourhood', type=int)
    parser.add_argument('--layers', help='number of layers', type=int)
    parser.add_argument('--in-joints', help='number of input joints', type=int, default=17)
    parser.add_argument('--out-joints', help='number of output joints', type=int, default=17)
    parser.add_argument('--dropout', help='dropout probability', type=float)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)

    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2, choices=[2, 3])
    parser.add_argument('--flip-data', help='train time flip', action='store_true')
    parser.add_argument('--output_file', type=str, default=None, help='Output file where save the informations pf the process')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint path to resume training from')
    parser.add_argument('--train_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)
    parser.add_argument('--test_set', type=str, default=None, help='Filename of the dataset', choices=["h36m", "humansc3d", "mpii"],required=True)

    try :
        args = parser.parse_args()
    except:
        parser.print_help()
        raise SystemExit

    return args


if __name__ == '__main__': 
    args = parse_args()

    datareader = data.DataReader()
    gt_trainset_all = datareader.real_read(args.train_set, "train")
    gt_testset_all = datareader.real_read(args.test_set, "test")
    
    mask = np.random.randint(0, 2, 10000).tolist()
    gt_trainset = [val for val, mask in zip(gt_trainset_all, mask) if mask == 1]

    mask = np.random.randint(0, 2, 100).tolist()
    gt_testset = [val for val, mask in zip(gt_testset_all, mask) if mask == 1]

    train_data, test_data = datareader.read_2d(gt_trainset, gt_testset, read_confidence=True if args.in_F == 3 else False)
    train_labels, test_labels = datareader.read_3d()

    if args.flip_data:
        train_data = data.flip_data(train_data)
        train_labels = data.flip_data(train_labels)

    # Impostiamo un valore fisso per il numero di epoche
    # Così possiamo testare il modello con diverse configurazioni di parametri
    # Random perchè sono troppe le possibilità
    possible_params = {
        'knn': [1, 2, 3, 4, 5],
        'batch_size': [128, 256, 512],
        #'decay_steps': [10000, 20000, 30000, 40000, 50000],
        #'decay_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': [1],
        'dropout': [0.1, 0.25, 0.3],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
        'regularization': [0, 1e-4, 5e-4]
    }

    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    
    saves_configurations = []

    for i in range(2):
        # Randomly select parameters
        selected_params = {key: np.random.choice(value) for key, value in possible_params.items()}
        
        # Check if selected_params are already tested
        if selected_params in saves_configurations:
            print(f"Parameters {selected_params} already tested, skipping...")
            i -= 1
            continue

        args.test_indices = str(10 + i)
        args.layers = selected_params['num_layers']
        args.dropout = selected_params['dropout']
        args.knn = selected_params['knn']
        params_help.update_parameters(args, params)

        #Remove knn from selected_params
        #selected_params.pop('knn')
        # Update params with selected parameters
        params.update(selected_params)
        print(f"Selected parameters for iteration {i+1}: {selected_params}")
        
        # Here you would call your training function with the updated params
        print(pprint.pformat(params))
        network = models_att.cgcnn(**params)
        try: 
            losses, t_step = network.fit(train_data, train_labels, test_data, test_labels)
            selected_params['losses'] = losses
            selected_params['t_step'] = t_step
            saves_configurations.append(selected_params)
        except KeyboardInterrupt:
            print('Training interrupted')
            selected_params['error'] = "Training interrupted"
            saves_configurations.append(selected_params)
        except Exception as e:
            print('Error during training: ', e)
            selected_params['losses'] = None
            selected_params['t_step'] = None
            selected_params['error'] = str(e)
            saves_configurations.append(selected_params)
            #raise SystemExit
    
    # Save the configurations to a file
    ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(ROOT_PATH, "experiment", 'saves_configurations.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(saves_configurations, indent=3, cls=NpEncoder))