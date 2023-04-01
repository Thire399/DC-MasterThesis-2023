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

model = models.alexnet(pretrained = False)
model.classifier[6] = nn.Linear(in_features=4096, out_features = 2, bias=True)

#model = models.resnet50(pretrained = False)
#model.fc = nn.Linear(in_features=2048, out_features = 2, bias=True)
#
#model = M.UNet(enc_chs = (3, 64, 128, 256, 512, 1024)
#               , dec_chs = (1024, 512, 256, 128, 64)
#               , num_class = 1
#               , df = 16384) # binary classification = 1.

#Data parameters
dataSet      = 'chest_xray'
datatype     = ''
os.makedirs('Data/Loss_chest_xray/test', exist_ok = True)
costumLabel  = '64x64Full'
dev = True
#model parameters
patience     = 10 #
delta        = 1e-6
epochs       = 400

learningRate = 1e-3 #add weight decay weight_decay=1e-5
optimizer    = optim.SGD(model.parameters(), lr = learningRate, momentum = 0.5)#optim.Adam(model.parameters(), lr = learningRate)
loss_Fun     = nn.CrossEntropyLoss()
batch_size   = 64
saveModel    = True
figSave      = False
####### PARAMETERS #######

####### Main Calls ########

def __main__():

        xTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainX.pt')
        yTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainY.pt')

        xVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValX.pt')
        yVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValY.pt')

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
                p, t, fscore = Loop.eval_model(model = model
                                , dataset = dataSet
                                , dev = dev
                                , val_Loader = val_Loader
                                , size = costumLabel)
                

        print('Accuracy on temp ValidationSet: {0}     --> (sum(Prediction = Target))/n_sampels'.format(np.sum([p == t])/t.shape[0]))

        parser.print_aggregate(log_dir= 'Data/Loss_' + dataSet + '/CarbonLogs')
#
#        import pandas as pd
#        import plotly.graph_objects as go
#
#        fig = go.Figure(data=go.Scatterpolar(r = [10.0, 0.33, epochs]
#                                               ,theta = ['fscore', "time", "epoch"],
#                                                fill='toself'
#                                             )
#                                         )
#        print(fscore)
#        fig.update_layout(
#                polar=dict(
#                radialaxis=dict(
#                visible=True
#                ),
#            ),
#            showlegend=False
#        )
#        fig.show()
#        #df = pd.DataFrame(dict(r = [fscore, 0.33, epochs]
#        #                        ,theta = ['fscore', "time", "epoch"]))
#        #fig = px.line_polar(df, r='r', theta='theta', line_close=True)
#        #fig.show()
        return None


__main__()


