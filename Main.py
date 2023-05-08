#Main
import Loop
import Models as M
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models
from carbontracker import parser

homeDir = os.getcwd() #Gets homeDirectory dynamically
print(f'Running at "{homeDir}"...')
os.chdir(homeDir)

####### PARAMETERS #######

#model = models.densenet169(pretrained = False)
#model.classifier = nn.Linear(in_features=1664, out_features = 1, bias=True)

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
synthetic = True
dataSet      = 'Alzheimer_MRI'
#dataSet      = 'chest_xray'
datatype     = 'DMAfterLR1K3000'
#'10PercentDistribution'
costumLabel  = 'DMAfterLR1K3000'#'SyntheticMRI128x128'#
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
highRes      = False
####### PARAMETERS #######

####### Main Calls ########
def sigmoid(x):
    #print(1/(1+torch.exp(-x)))
    return 1 / (1+torch.exp(-x))

def __main__():
        print('Starting...')
        if synthetic:
            xTrain = torch.load(f'Data/Synthetic_{dataSet}/' + datatype + 'X.pt')
            yTrain = torch.load(f'Data/Synthetic_{dataSet}/' + datatype + 'Y.pt')
            with torch.no_grad():
                xTrain = xTrain.repeat(1,3,1,1).sigmoid_()
        elif highRes:
            import Data_processing as DP
            print('High resulotion...\nGetting file names...')
            files = DP.GetFileNames('Data/Proccesed/chest_xray/train/')
            train_set = DP.ChestXrayDataset(files = files)
            train_Loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 0)

        else:
            xTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainX.pt')
            yTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainY.pt')
            xTrain = xTrain.repeat(1, 3, 1, 1) # only for pretrained model
        if dataSet == 'chest_xray':
            if not highRes:
                xVal = torch.load('Data/Proccesed/'+ dataSet + '/tempValX.pt')
                yVal = torch.load('Data/Proccesed/'+ dataSet + '/tempValY.pt')
                xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model
            else:
                files = DP.GetFileNames('Data/Proccesed/chest_xray/temporaryVal/')
                val_Set = DP.ChestXrayDataset(files = files)
                val_Loader = torch.utils.data.DataLoader(val_Set,
                                                        batch_size = batch_size,
                                                        shuffle = True,
                                                        num_workers = 0)
                 
        else:
            xVal = torch.load('Data/Proccesed/'+ dataSet + '/ValX.pt')
            yVal = torch.load('Data/Proccesed/'+ dataSet + '/ValY.pt')
            xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model
        
        if not highRes:
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


