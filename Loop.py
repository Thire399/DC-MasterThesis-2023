import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import Models as M
from torchvision import models
from carbontracker.tracker import CarbonTracker

os.chdir('/home/thire399/Documents/DC-MasterThesis-2023')

####### PARAMETERS #######

#resnet50(weights= ResNet50_Weights.DEFAULT) # DEFAULT is in this case alias for IMAGENET1K_V2
#resnet50(weights = "IMAGENET1K_V2")
#resnet50(pretrained = True)
#resnet50(True)
#model = models.resnet18(pretrained = True)
#model.fc = nn.Linear(in_features=2048, out_features = 2, bias=True)

model = M.UNet(enc_chs = (3, 64, 128, 256, 512, 1024)
               , dec_chs = (1024, 512, 256, 128, 64)
               , num_class = 1) # binary classification = 1.
dataSet = 'chest_xray'
patience = 10 #
delta = 1e-4
epochs     = 20

learningRate = 1e-6
optimizer = optim.Adam(model.parameters(), lr = learningRate)
loss_Fun   = nn.CrossEntropyLoss()
batch_size = 64
saveModel = True
####### PARAMETERS #######

def TrainLoop(
            train_Loader
            , val_Loader
            , model     = model
            , patience  = patience
            , delta     = delta
            , epochs    = epochs
            , optimizer = optimizer
            , loss_Fun  = loss_Fun
            , modelSave = saveModel
            ):
    tracker = CarbonTracker(epochs=epochs)
    #### -- Set up -- ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    # if device == 'cpu':
        # print('Not gonna run on the CPU')
        # return None, None, None


    mkPathLoss = 'Data/Loss_' + dataSet
    os.makedirs(mkPathLoss , exist_ok = True)
    os.makedirs(mkPathLoss + '/Figs' , exist_ok = True)

    now = time.strftime("%Y%m%d-%H%M%S") #save file as current time stamp - better format to save file?

    early_stopping = M.EarlyStopping( patience = patience,
                            verbose = True,
                            delta   = delta,
                            # path    = mkPathLoss + 'Figs/' + model._get_name() + now
                            )
    train_Loss = []
    val_Loss = []
    batchTrain_loss = []
    batchVal_loss   = []
    #### -- Set up -- ####

    #### -- Main loop -- ####
    model.to(device)
    for epoch in range(epochs):
        tracker.epoch_start()
        batchTrain_loss = []
        batchVal_loss = []

        model.train()
        #Train Data
        for batch, (data, target) in enumerate(train_Loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            out = model(data.to(device))
            loss = loss_Fun(out, (target).type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            batchTrain_loss.append(loss.item())
            if batch % 2 == 0:
                print('Train Epoch [{}/{}]: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, epochs, batch * len(data), len(train_Loader.dataset),
                    100. * batch / len(train_Loader),
                    np.mean(batchTrain_loss)))
        train_Loss.append(np.mean(batchTrain_loss))

        # Val Data.
        model.eval()
        for batch, (data, target) in enumerate(val_Loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            out = model(data.to(device))
            loss = loss_Fun(out, (target).type(torch.LongTensor).to(device))
            batchVal_loss.append(loss.item())
            if batch % 2 == 0: #For printing
                print(4*' ', '===> Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch * len(data), len(val_Loader.dataset),
                    100. * batch / len(val_Loader),
                    np.mean(batchVal_loss)))

        tracker.epoch_end()
        temp_ValLoss = np.mean(batchVal_loss)
        val_Loss.append(temp_ValLoss)

        early_stopping(temp_ValLoss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            #tracker.stop()
            break
        else:
            continue
    if not early_stopping.early_stop:
        tracker.stop()
    #### -- Save info -- ####
    if modelSave == True:

        t_Loss = torch.tensor(train_Loss)
        v_Loss = torch.tensor(val_Loss)

        torch.save(t_Loss, f = mkPathLoss + '/train_loss'+ model._get_name()  + now + '.pt') # add name
        torch.save(t_Loss, f = mkPathLoss + '/val_loss'  + model._get_name() + now + '.pt')   # add name
        torch.save(model.state_dict(), mkPathLoss + '/model'+ model._get_name() + now + '.pt' ) # saves model.
    return None

####### Main Calls ########

xTrain = torch.load('Data/Proccesed/'+ dataSet +'/trainX.pt')
yTrain = torch.load('Data/Proccesed/'+ dataSet +'/trainY.pt')

xVal = torch.load('Data/Proccesed/'+ dataSet +'/valX.pt')
yVal = torch.load('Data/Proccesed/'+ dataSet +'/valY.pt')

xTrain = xTrain.repeat(1, 3, 1, 1) # only for pretrained model
xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model

train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,#
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 0)

val_Set = torch.utils.data.TensorDataset(xVal, yVal)
val_Loader = torch.utils.data.DataLoader(val_Set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 0)
TrainLoop(train_Loader = train_Loader
          , val_Loader = val_Loader
          , model    = model
          , patience = patience
          , delta    = 1e-4
          , epochs = epochs
          , optimizer = optimizer
          , loss_Fun = loss_Fun
          , modelSave = saveModel
          )