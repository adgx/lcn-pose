import numpy as np
from tools import params_help, data
from network import  models_att # models_attr2 as
import os
import pprint
import numpy as np
from train import parse_args

from multiple_trains import write_to_file


if __name__ == '__main__': 
    args = parse_args()

    datareader = data.DataReader()
    gt_trainset_all = datareader.real_read(args.train_set, "train")
    gt_testset_all = datareader.real_read(args.test_set, "test")
    
    gt_trainset = data.get_subset(gt_trainset_all, subset_size=args.subset, mode="camera")
    gt_testset = data.get_subset(gt_testset_all, subset_size=args.subset, mode="camera")

    train_data, test_data, train_labels, test_labels = None, None, None, None
    train_data, test_data = datareader.read_2d(gt_trainset, gt_testset)
    train_labels, test_labels = datareader.read_3d()

    if args.flip_data:
        train_data_flip = data.flip_data(train_data)
        train_labels_flip = data.flip_data(train_labels)

    if args.translate_data:
        translation_factor = args.translation_factor
        if translation_factor < 0:
            raise ValueError("Translation factor must be non-negative")
        translation = np.random.uniform(-translation_factor, translation_factor)
        train_data_translate = data.translation_data(train_data, translation)
        train_labels_translate = data.translation_data(train_labels, translation)

    if args.rotation_data:
        train_data_rotated = data.rotate_data(train_data)
        train_labels_rotated = data.rotate_data(train_labels)
    

    if args.flip_data:
        train_data = np.concatenate((train_data, train_data_flip), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_flip), axis=0)

    if args.translate_data:
        train_data = np.concatenate((train_data, train_data_translate), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_translate), axis=0)

    if args.rotation_data:
        train_data = np.concatenate((train_data, train_data_rotated), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_rotated), axis=0)

    # Impostiamo un valore fisso per il numero di epoche
    # Così possiamo testare il modello con diverse configurazioni di parametri
    # Random perchè sono troppe le possibilità
    possible_params = {
        'knn': [1, 3, 5],
        'batch_size': [128, 256, 512],
        #'decay_steps': [10000, 20000, 30000, 40000, 50000],
        #'decay_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': [1, 3, 5],
        'dropout': [0.25, 0.35, 0.4],
        'learning_rate': [1e-3, 1e-4],
        'regularization': [5e-4, 5e-2, 5e-3],

    }

    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    
    saves_configurations = []

    # Load existing configurations if available
    ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(ROOT_PATH, "experiment", 'saves_configurations.csv')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header line
        for line in lines:
            parts = line.strip().split(',')
            config = {
                'knn': int(parts[0]),
                'batch_size': int(parts[1]),
                'num_layers': int(parts[2]),
                'dropout': float(parts[3]),
                'learning_rate': float(parts[4]),
                'regularization': float(parts[5]),
            }
            saves_configurations.append(config)

    for i in range(args.n_test):
        # Randomly select parameters
        selected_params = {key: np.random.choice(value) for key, value in possible_params.items()}
        
        # Check if selected_params are already tested
        if selected_params in saves_configurations:
            print(f"Parameters {selected_params} already tested, skipping...")
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
            selected_params['mean_loss'] = np.mean(losses)
            selected_params['std_loss'] = np.std(losses)
            selected_params['validation_loss'] = losses[-1] if losses else None
            selected_params['t_step'] = t_step
            selected_params["best_loss"] = np.min(losses) if losses else None
            write_to_file(selected_params, config_file)
        except KeyboardInterrupt:
            print('Training interrupted')
            selected_params['error'] = "Training interrupted"
            write_to_file(selected_params, config_file)
        except Exception as e:
            print('Error during training: ', e)
            selected_params['losses'] = None
            selected_params['t_step'] = None
            write_to_file(selected_params, config_file)
            #raise SystemExit
