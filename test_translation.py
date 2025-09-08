import numpy as np
import matplotlib.pyplot as plt
from tools import data


datareader = data.DataReader()
gt_trainset = datareader.real_read("humansc3d", "train")
gt_valset = datareader.real_read("humansc3d", "val")

train_data, val_data = datareader.read_2d(gt_trainset, gt_valset)
train_data = train_data[10:200]

#Translate Data
translate_data = data.translation_data(train_data, translation_factor=0.5)

untranslate_data = data.untranslation_data(translate_data, translation_factor=0.5)

train_data = train_data.reshape(5, 17, -1)
translate_data = translate_data.reshape(5, 17, -1)
untranslate_data = untranslate_data.reshape(5, 17, -1)

#Check if untranslate_data and train_data are equal
if np.array_equal(untranslate_data, train_data):
    print("Untranslated data is equal to original training data.")
else:
    print("Untranslated data is NOT equal to original training data.")

for i in range(5):

    #Create a plot
    ax = plt.figure()
    ax = plt.axes()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    ax.set_aspect('equal')

    # Plot the data
    for j in range(17):
        ax.plot(train_data[i][j, 0], translate_data[i][j, 1], 'o', markersize=8)
        ax.plot(translate_data[i][j, 0], translate_data[i][j, 1], 'o', markersize=8)

    ax.legend(["Original", "translated"])
    
    plt.title("2D Joint Coordinates: Original vs Translated")
    plt.legend(["Original", "Translated"])
    plt.show()


def test():

    #Reshape them
    reshape_data = np.array(data_item).reshape(17, -1) # [17, 2]

    # Translate data
    translate_data = reshape_data.copy()
    for i in range(len(translate_data)):
        #Sum values of each joint's x and y coordinates
        translate_data[i, 0] += 0.5
        translate_data[i, 1] += 0.5