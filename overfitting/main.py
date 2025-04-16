import os
import argparse
import numpy as np 
import plotly.express as px
from functions import read_json
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_PATH = os.path.join(ROOT_PATH, "data", "training")
VALIDATION_PATH = os.path.join(ROOT_PATH, "data", "validation")

PATH_AVG_LOSS = os.path.join(TRAINING_PATH, "avg_loss")
PATH_LOSS = os.path.join(TRAINING_PATH, "loss")
PATH_AVG_LOSS_TOTALE = os.path.join(TRAINING_PATH, "avg_loss_total")
PATH_LOSS_TOTALE = os.path.join(TRAINING_PATH, "loss_total")

PATH_VAL_LOSS = os.path.join(VALIDATION_PATH, "loss")

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', type=str, default="hm_2L", help='Name of the dataset', choices=["hm_2L", "hm_3L", "human_2L", "human_3L"],required=True)

    try :
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        raise SystemExit

if __name__ == "__main__":
    
    #args = parse_args()
    fig = make_subplots(rows=2, cols=2, subplot_titles=("H3.6M 2 Layer", "H3.6M 3 Layer", "HumanScan3D 2 Layer", "HumanScan3D 3 Layer"))

    choices = ["hm_2L", "hm_3L", "human_2L", "human_3L"]
    for idx, plot in enumerate(choices):
        dataset = choices[idx]
        print(f"Processing dataset: {dataset}")
        avg_loss_json = os.path.join(PATH_AVG_LOSS, dataset + ".json")
        if not os.path.exists(avg_loss_json):
            raise FileNotFoundError(f"File {avg_loss_json} does not exist.")
        loss_json = os.path.join(PATH_LOSS, dataset + ".json")
        avg_total_loss = os.path.join(PATH_AVG_LOSS_TOTALE, dataset + ".json")
        if not os.path.exists(avg_loss_json):
            raise FileNotFoundError(f"File {avg_loss_json} does not exist.")
        total_loss = os.path.join(PATH_LOSS_TOTALE, dataset + ".json")
        if not os.path.exists(avg_loss_json):
            raise FileNotFoundError(f"File {avg_loss_json} does not exist.")
        val_loss = os.path.join(PATH_VAL_LOSS, dataset + ".json")
        if not os.path.exists(avg_loss_json):
            raise FileNotFoundError(f"File {avg_loss_json} does not exist.")
        
        avg_loss_json = read_json(avg_loss_json)
        loss_json = read_json(loss_json)
        avg_total_loss = read_json(avg_total_loss)
        total_loss = read_json(total_loss)
        val_loss = read_json(val_loss)

        lowest_val_loss = np.min(val_loss[:, 2])
        lowest_val_loss_epoch = np.argmin(val_loss[:, 2])
        lowest_val_loss_epoch = val_loss[lowest_val_loss_epoch, 1]
        print(f"Lowest validation loss: {lowest_val_loss} at epoch {lowest_val_loss_epoch}")
        
        i = idx // 2 + 1
        j = idx % 2 + 1
        fig.add_trace(
            go.Scatter(x=avg_loss_json[:, 1], y=avg_loss_json[:, 2], mode='lines', name='Training Average Loss'),
            row=i, col=j
        )
        fig.add_trace(
            go.Scatter(x=loss_json[:, 1], y=loss_json[:, 2], mode='lines', name='Training Loss'),
            row=i, col=j
        )
        fig.add_trace(go.Scatter(x=loss_json[:, 1], y=loss_json[:, 2], mode='lines', name='Training Loss', showlegend=False, legendgroup="t1"), row=i, col=j)
        fig.add_trace(go.Scatter(x=avg_total_loss[:, 1], y=avg_total_loss[:, 2], mode='lines', name='Training Total Average Loss',showlegend=False, legendgroup="t2"), row=i, col=j)
        fig.add_trace(go.Scatter(x=total_loss[:, 1], y=total_loss[:, 2], mode='lines', name='Training Total Loss', showlegend=False, legendgroup="t3"),row=i, col=j)
        fig.add_trace(go.Scatter(x=val_loss[:, 1], y=val_loss[:, 2], mode='lines', name='Validation Loss', showlegend=False, legendgroup="t4"), row=i, col=j)
        fig.add_trace(go.Scatter(x=[lowest_val_loss_epoch], y=[lowest_val_loss], mode='markers', name='Lowest Validation Loss', marker=dict(color='red', size=10), showlegend=False, legendgroup="t5"), row=i, col=j) 
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(showlegend=True)

    fig.show()

    



    