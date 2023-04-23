import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import os

os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')

dataSet = 'Alzheimer_MRI'
customLabel = 'Test'
x = torch.load(f'Data/Synthetic_{dataSet}/{customLabel}X.pt')
y = torch.load(f'Data/Synthetic_{dataSet}/{customLabel}Y.pt')

c0 = random.sample(range(0, 101), 5)
c1 = random.sample(range(100, 201), 5)

fig, ax = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
    ax[0][i].imshow(x.detach().numpy()[c0[i]][0], cmap = 'gray')
    ax[0][i].set_title(f'Image: {c0[i]}\n(class 0)')
    ax[0][i].axis('off')
    ax[1][i].imshow(x.detach().numpy()[c1[i]][0], cmap = 'gray')
    ax[1][i].set_title(f'Image: {c1[i]}\n(class 1)')
    ax[1][i].axis('off')
plt.show()


dataSet = 'Alzheimer_MRI'
customLabel = 'Before'
x = torch.load(f'Data/Synthetic_{dataSet}/{customLabel}X.pt')
y = torch.load(f'Data/Synthetic_{dataSet}/{customLabel}Y.pt')

c0 = random.sample(range(0, 101), 5)
c1 = random.sample(range(100, 201), 5)

fig, ax = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
    ax[0][i].imshow(x.detach().numpy()[c0[i]][0], cmap = 'gray')
    ax[0][i].set_title(f'Image: {c0[i]}\n(class 0)')
    ax[0][i].axis('off')
    ax[1][i].imshow(x.detach().numpy()[c1[i]][0], cmap = 'gray')
    ax[1][i].set_title(f'Image: {c1[i]}\n(class 1)')
    ax[1][i].axis('off')
plt.show()