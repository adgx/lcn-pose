import os
import argparse
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_PATH = os.path.join(ROOT_PATH, "data", "hyperparameter" , "training")
VALIDATION_PATH = os.path.join(ROOT_PATH, "data", "hyperparameter" , "validation")

def add_graphs(fig):
    loss_csv = os.path.join(TRAINING_PATH, "test" + str(13) + "_summaries.csv")
    if not os.path.exists(loss_csv):
        raise FileNotFoundError(f"File {loss_csv} does not exist.")
    val_csv = os.path.join(VALIDATION_PATH, "test" + str(13) + "_summaries.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"File {val_csv} does not exist.")
        
       
    loss_json = read_csv(loss_csv)
    val_loss = read_csv(val_csv)

    fig.add_trace(go.Scatter(x=loss_json[:, 1], y=loss_json[:, 2], mode='lines', name='Training Loss', showlegend=False, legendgroup="t1"))
    fig.add_trace(go.Scatter(x=val_loss[:, 1], y=val_loss[:, 2], mode='lines', name='Validation Loss', showlegend=False, legendgroup="t4"))
    fig.add_trace(go.Scatter(x=[lowest_val_loss_epoch], y=[lowest_val_loss], mode='markers', name='Lowest Validation Loss', marker=dict(color='red', size=10), showlegend=False, legendgroup="t5")) 
    

    

if __name__ == "__main__":
        
    def read_csv(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        return np.genfromtxt(file_path, delimiter=',', skip_header=1)
    

    # idx of the tensorboard summaries to plot
    idx = 16

    # Single plot for a specific idx
    print(f"Processing dataset: {idx}")



    fig = go.Figure()
        
    loss_csv = os.path.join(TRAINING_PATH, "test" + str(idx) + "_summaries.csv")
    if not os.path.exists(loss_csv):
        raise FileNotFoundError(f"File {loss_csv} does not exist.")
    val_csv = os.path.join(VALIDATION_PATH, "test" + str(idx) + "_summaries.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"File {val_csv} does not exist.")
        
       
    loss_json = read_csv(loss_csv)
    val_loss = read_csv(val_csv)

    # Compute the difference between the training and validation loss
    diff_loss = val_loss[:, 2][:50] - loss_json[:, 2][:50]
    # Print the difference
    print(f"Difference between training and validation loss: {diff_loss}")

    #Write the difference to a file
    diff_loss_file = os.path.join(ROOT_PATH, "data", "hyperparameter", "diff_loss_" + str(idx) + ".txt")
    with open(diff_loss_file, "w") as f:
        for i in range(len(diff_loss)):
            f.write(f"{i}, {diff_loss[i]}\n")

    # Il numero di data è uguale al numero di steps
    num_epochs = 100
    num_steps = loss_json.shape[0] 
    
    # 
    tmp = num_steps // num_epochs
    #Take a value every tmp steps
    loss_json = loss_json[::tmp, :]
    val_loss = val_loss[::tmp, :]


    lowest_val_loss = np.min(val_loss[:, 2])
    lowest_val_loss_epoch = np.argmin(val_loss[:, 2])
    lowest_val_loss_epoch = val_loss[lowest_val_loss_epoch, 1]
    print(f"Lowest validation loss: {lowest_val_loss} at epoch {lowest_val_loss_epoch}")
        
    i = idx // 2 + 1
    j = idx % 2 + 1
       
    fig.add_trace(go.Scatter(x=loss_json[:, 1], y=loss_json[:, 2], mode='lines', name='Training Loss', showlegend=False, legendgroup="t1"))
    fig.add_trace(go.Scatter(x=val_loss[:, 1], y=val_loss[:, 2], mode='lines', name='Validation Loss', showlegend=False, legendgroup="t4"))
    fig.add_trace(go.Scatter(x=[lowest_val_loss_epoch], y=[lowest_val_loss], mode='markers', name='Lowest Validation Loss', marker=dict(color='red', size=10), showlegend=False, legendgroup="t5")) 
    
    # Get 
    add_graphs(fig)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(showlegend=True)

    fig.show()

    

