import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import Models as M

os.chdir('/Users/thire/Documents/School/DC-MasterThesis-2023')

####### PARAMETERS #######
model = M.UNet
dataSet = 'chest_xray'
patience = 100 # 
delta = ''
LossPlotPath = ''
epochs     = 200
learningRate = 0.2
optimizer = optim.Adam(model.parameters(), lr = 0.2)
loss_Fun   = nn.CrossEntropyLoss
batch_size = 64
####### PARAMETERS #######

def TrainLoop(
            train_Loader
            , val_Loader
            , model     = model
            , patience  = patience
            , delta     = delta
            , path      = ''
            , epochs    = epochs
            , optimizer = optimizer
            , loss_Fun  = loss_Fun
            ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    # if device == 'cpu':
        # print('Not gonna run on the CPU')
        # return None, None, None
    early_stopping = M.EarlyStopping( patience = patience, 
                            verbose=True,
                            delta = delta,
                            path = path)
    Train_loss = []
    Val_loss = []
    batchTrain_loss = []
    batchVal_loss   = []


    for epoch in epochs:
        batchTrain_loss = []
        batchVal_loss = []
        
        model.train()
        #Train Data
        for batch, (data, target) in enumerate(train_Loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            out = model(data.to(device))
            loss = loss_Fun(out, (target).float().to(device))
            loss.backward() 
            optimizer.step()
            batchTrain_loss.append(loss.item())
            
        # Val Data.
        model.test()
        for batch, (data, target) in enumerate(val_Loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            out = model(data.to(device))
            loss = loss_Fun(out, (target).float().to(device))
            batchVal_loss.append(loss.item())


        temp = np.mean(batchVal_loss)
        Val_loss.append(temp)

        early_stopping(temp, model)
        
        if early_stopping.early_stop:
            print("Early stopping") 
            break
        else:
            continue

    return None
    
xTrain = torch.load('Data/Proccesed/'+ dataSet +'/trainX.pt')
yTrain = torch.load('Data/Proccesed/'+ dataSet +'/trainY.pt')

xVal = torch.load('Data/Proccesed/'+ dataSet +'/valX.pt')
yVal = torch.load('Data/Proccesed/'+ dataSet +'/valY.pt')
train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            num_workers = 0)

val_Set = torch.utils.data.TensorDataset(xVal, yVal)
val_Loader = torch.utils.data.DataLoader(val_Set,
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            num_workers = 0)

for batch, (data, target) in enumerate(train_Loader, 1):
    print(data.size())
    break