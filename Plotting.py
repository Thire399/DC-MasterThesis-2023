import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import os

os.chdir(os.getcwd())
def sigmoid(x):
    #print(1/(1+torch.exp(-x)))
    return 1 / (1+torch.exp(-x))
dataSet = 'Alzheimer_MRI'
customLabel = 'Before'
x = torch.load(f'Data/Synthetic_{dataSet}/GM{customLabel}X.pt')
y = torch.load(f'Data/Synthetic_{dataSet}/GM{customLabel}Y.pt')

c0 = random.sample(range(0, 5), 5)
c1 = random.sample(range(5, 10), 5)

fig, ax = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
    ax[0][i].imshow(x.detach().numpy()[c0[i]][0], cmap = 'gray')
    ax[0][i].set_title(f'Image: {c0[i]}\n(class 0)')
    ax[0][i].axis('off')
    ax[1][i].imshow(x.detach().numpy()[c1[i]][0], cmap = 'gray')
    ax[1][i].set_title(f'Image: {c1[i]}\n(class 1)')
    ax[1][i].axis('off')
plt.show()


customLabel = 'After'
x = torch.load(f'Data/Synthetic_{dataSet}/GM{customLabel}X.pt')
y = torch.load(f'Data/Synthetic_{dataSet}/GM{customLabel}Y.pt')

c0 = random.sample(range(0, 5), 5)
c1 = random.sample(range(5, 10), 5)

fig, ax = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
    ax[0][i].imshow(sigmoid(x).detach().numpy()[c0[i]][0], cmap = 'gray')
    ax[0][i].set_title(f'Image: {c0[i]}\n(class 0)')
    ax[0][i].axis('off')
    ax[1][i].imshow(sigmoid(x).detach().numpy()[c1[i]][0], cmap = 'gray')
    ax[1][i].set_title(f'Image: {c1[i]}\n(class 1)')
    ax[1][i].axis('off')
plt.savefig('temporary.png', bbox_inches = 'tight', dpi = 400)
plt.show()