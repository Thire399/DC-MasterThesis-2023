import numpy as np
import matplotlib.pyplot as plt
import torch
import os

os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')

dataSet = 'chest_xray'
#64x64FullUNet20230329-015602.png
train = torch.load('Data/Loss_chest_xray/train_loss64x64FullUNet20230329-015602.pt').tolist()
val = torch.load('Data/Loss_chest_xray/val_loss64x64FullUNet20230329-015602.pt').tolist()


plt.plot(range(len(train)), train, label = 'Train')
plt.plot(range(len(val)), val, label = 'val')
plt.axvline(val.index(min(val)), linestyle='--', color='r',label='Early Stopping Checkpoint')
legend = plt.legend(loc = 'upper right', frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('grey')
frame.set_edgecolor('black')
plt.xlabel ('Epoch')
plt.ylabel('Loss')
plt.title('UNet64x64: Loss per epoch')
plt.grid()
plt.savefig('Data/Loss_chest_xray/Figs/UNet64x64.png', dpi = 400, bbox_inches = 'tight' )
plt.show()