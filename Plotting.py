import numpy as np
import matplotlib.pyplot as plt
import torch
import os

os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')

dataSet = 'chest_xray'

train = torch.load('Data/Loss_chest_xray/train_lossResNet20230309-211533.pt').tolist()
val = torch.load('Data/Loss_chest_xray/val_lossResNet20230309-211533.pt').tolist()

plt.plot(range(len(train)), train, label = 'Train')
plt.plot(range(len(val)), val, label = 'val')
plt.axvline(val.index(min(val)), linestyle='--', color='r',label='Early Stopping Checkpoint')
legend = plt.legend(loc = 'upper right', frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('grey')
frame.set_edgecolor('black')
plt.xlabel ('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.show()