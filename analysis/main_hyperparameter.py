import os
import argparse
import numpy as np 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_PATH = os.path.join(ROOT_PATH, "data", "hyperparameter" , "training")
VALIDATION_PATH = os.path.join(ROOT_PATH, "data", "hyperparameter" , "validation")

if __name__ == "__main__":
    
    #count how many files are in the directory
    def count_files_in_directory(directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    
    def read_csv(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        return np.genfromtxt(file_path, delimiter=',', skip_header=1)
    


    #args = parse_args()
    total = count_files_in_directory(TRAINING_PATH)
    if total == 0:
        raise FileNotFoundError(f"No files found in directory {TRAINING_PATH}.")
    
    rows = total // 2 + (total % 2 > 0)
    print(f"Total files found: {total}, Rows needed: {rows}")
    # Create subplots
    subplot_titles = [f"Test {i+1}" for i in range(total)]
    fig = make_subplots(
       cols=2,
       rows=rows,
        
    )

    for idx, plot in enumerate(subplot_titles):
        print(f"Processing dataset: {idx}")
        
        loss_csv = os.path.join(TRAINING_PATH, "test" + str(idx) + "_summaries.csv")
        if not os.path.exists(loss_csv):
            raise FileNotFoundError(f"File {loss_csv} does not exist.")
        val_csv = os.path.join(VALIDATION_PATH, "test" + str(idx) + "_summaries.csv")
        if not os.path.exists(val_csv):
            raise FileNotFoundError(f"File {val_csv} does not exist.")
        
       
        loss_json = read_csv(loss_csv)
        val_loss = read_csv(val_csv)

        lowest_val_loss = np.min(val_loss[:, 2])
        lowest_val_loss_epoch = np.argmin(val_loss[:, 2])
        lowest_val_loss_epoch = val_loss[lowest_val_loss_epoch, 1]
        print(f"Lowest validation loss: {lowest_val_loss} at epoch {lowest_val_loss_epoch}")
        
        i = idx // 2 + 1
        j = idx % 2 + 1
       
        fig.add_trace(go.Scatter(x=loss_json[:, 1], y=loss_json[:, 2], mode='lines', name='Training Loss', showlegend=False, legendgroup="t1"), row=i, col=j)
        fig.add_trace(go.Scatter(x=val_loss[:, 1], y=val_loss[:, 2], mode='lines', name='Validation Loss', showlegend=False, legendgroup="t4"), row=i, col=j)
        fig.add_trace(go.Scatter(x=[lowest_val_loss_epoch], y=[lowest_val_loss], mode='markers', name='Lowest Validation Loss', marker=dict(color='red', size=10), showlegend=False, legendgroup="t5"), row=i, col=j) 

        # Compute the difference between training and validation loss
        loss_diff = val_loss[:, 2] - loss_json[:, 2]
        # Print the minimum difference
        min_loss_diff = np.min(loss_diff) 
        min_loss_diff_epoch = np.argmin(loss_diff)
        min_loss_diff_epoch = loss_json[min_loss_diff_epoch, 1]
        print(f"Minimum loss difference: {min_loss_diff} at epoch {min_loss_diff_epoch}")

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(showlegend=True)

    fig.show()

    

