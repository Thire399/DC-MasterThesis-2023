#Main
import numpy as np
import Loop
import Models as M
import torch
import torch.nn as nn
import torch.optim as optim
import os
import plotly.express as px
from torchvision import models
from carbontracker import parser

os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')

####### PARAMETERS #######

#model = models.alexnet(pretrained = False)
#model.classifier[6] = nn.Linear(in_features=4096, out_features = 1, bias=True)
#model = models.efficientnet_v2_s(pretrained = False)
#model.classifier[1] = nn.Linear(in_features=1280, out_features = 1, bias=True)
#model = models.inception_v3(pretrained = False)
#model.fc = nn.Linear(in_features=2048, out_features = 1, bias=True)

model = models.resnet50(pretrained = False)
model.fc = nn.Linear(in_features = 2048, out_features = 1, bias = True)

#model.fc.add_module('Sigmoid', nn.Sigmoid())
#model = M.UNet(enc_chs = (3, 64, 128, 256, 512, 1024)
#               , dec_chs = (1024, 512, 256, 128, 64)
#               , num_class = 1
#               , df = 16384) # binary classification = 1.

#Data parameters
dataSet      = 'Alzheimer_MRI'
#dataSet      = 'chest_xray'
datatype     = ''
#'10PercentDistribution'
costumLabel  = '128x128Full'#
#costumLabel = '64x6410PercentDistribution'

dev = False
#model parameters
patience     = 10 #
delta        = 1e-4
epochs       = 400

learningRate = 1e-3
optimizer    = optim.SGD(model.parameters(), lr = learningRate, momentum = 0.9)
#optimizer    =  optim.Adam(model.parameters(), lr = learningRate)
loss_Fun     = nn.BCEWithLogitsLoss()
batch_size   = 32
saveModel    = True
figSave      = True
####### PARAMETERS #######

####### Main Calls ########

def __main__():

        xTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainX.pt')
        yTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainY.pt')
        if dataSet == 'chest_xray':
            xVal = torch.load('Data/Proccesed/'+ dataSet + '/tempValX.pt')
            yVal = torch.load('Data/Proccesed/'+ dataSet + '/tempValY.pt')
        else:
            xVal = torch.load('Data/Proccesed/'+ dataSet + '/ValX.pt')
            yVal = torch.load('Data/Proccesed/'+ dataSet + '/ValY.pt')
        xTrain = xTrain.repeat(1, 3, 1, 1) # only for pretrained model
        xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model
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

        # p = prediction, t = target
        Loop.TrainLoop(train_Loader = train_Loader
                , val_Loader = val_Loader
                , model    = model
                , patience = patience
                , delta    = 1e-4
                , epochs = epochs
                , optimizer = optimizer
                , loss_Fun = loss_Fun
                , modelSave = saveModel
                , figSave = figSave
                , dataSet = dataSet
                , costumLabel = costumLabel
                , dev = dev
                )
        if saveModel:
                fscore, pred, _ = Loop.eval_model(model = model
                                , dataset = dataSet
                                , dev = dev
                                , val_Loader = val_Loader
                                , size = costumLabel)
            #print('Accuracy on temp ValidationSet: {0}     --> (sum(Prediction = Target))/n_sampels'.format(np.sum([p == t])/t.shape[0]))        

        
        if dev:
            parser.print_aggregate(log_dir= 'Data/Loss_' + dataSet + '/test/CarbonLogs') 
        else:
            parser.print_aggregate(log_dir= 'Data/Loss_' + dataSet + '/CarbonLogs')
        return None


__main__()


