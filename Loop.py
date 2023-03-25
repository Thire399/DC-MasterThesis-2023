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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')

####### PARAMETERS #######

model = models.alexnet(pretrained = False)
model.classifier[6] = nn.Linear(in_features=4096, out_features = 2, bias=True)

#model = models.resnet50(pretrained = False)
#model.fc = nn.Linear(in_features=1024, out_features = 2, bias=True)
#
#model = M.UNet(enc_chs = (3, 64, 128, 256, 512, 1024)
#               , dec_chs = (1024, 512, 256, 128, 64)
#               , num_class = 1
#               , df = 4096) # binary classification = 1.

#Data parameters
dataSet      = 'chest_xray'
datatype     = ''
os.makedirs('Data/Loss_chest_xray/test', exist_ok = True)
costumLabel  = '64x64Full'
dev = True
#model parameters
patience     = 10 #
delta        = 1e-7
epochs       = 200

learningRate = 1e-5 #add weight decay weight_decay=1e-5
optimizer    = optim.SGD(model.parameters(), lr = learningRate, momentum = 0.5)#optim.Adam(model.parameters(), lr = learningRate)
loss_Fun     = nn.CrossEntropyLoss()
batch_size   = 64
saveModel    = True
figSave      = False
####### PARAMETERS #######

# Maybe move to another file?
def PRAUC(pred, Target, ep):
    '''
    Takes in output from model and Target labels,
    and an epsilong to avoid div by 0
    Calculates the precision and recall for the given model over multiple 
    thressholds, by using the sklearn implementation
    https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
    '''
#    pred = pred.contiguous().view(-1)
#    pred = pred.cpu().detach().numpy()
#    Target = Target.contiguous().view(-1)
#    Target = Target.cpu().detach().numpy()
    
    #rewrite into a function.
    P, R, T = precision_recall_curve(y_true = Target, probas_pred = pred)
    fscore = (2 * P * R) / (P + R + ep)

    # locate the index of the largest f score
    ix = np.argmax(fscore)
    bestT = T[ix]
    print('Best Threshold=%f, F-Score=%.3f' % (T[ix], fscore[ix]))

    no_skill = len(Target[Target == 1]) / len(Target)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(R, P, marker='.', label = 'Logistic')
    legend = plt.legend(loc = 'upper right', frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('grey')
    frame.set_edgecolor('black')
    plt.xlabel ('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.show()
    return P[ix], R[ix], bestT

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
            , figSave = figSave
            , dev = dev
            ):
    #tracker = CarbonTracker(epochs=epochs)
    #### -- Set up -- ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    if device == 'cpu':
         print('Not gonna run on the CPU')
         return None

    mkPathLoss = 'Data/Loss_' + dataSet
    os.makedirs(mkPathLoss , exist_ok = True)
    os.makedirs(mkPathLoss + '/Figs' , exist_ok = True)

    now = time.strftime("%Y%m%d-%H%M%S") #save file as current time stamp - better format to save file?
    if dev:
        early_stopping = M.EarlyStopping( patience = patience,
                                verbose = True,
                                delta   = delta,
                                path    =  mkPathLoss + '/test' + '/model' + costumLabel + model._get_name() + now,
                                saveModel = modelSave)
    else:
        early_stopping = M.EarlyStopping( patience = patience,
                        verbose = True,
                        delta   = delta,
                        path    =  mkPathLoss + '/model' + costumLabel + model._get_name() + now,
                        saveModel = modelSave)

        
    train_Loss = []
    val_Loss = []
    batchTrain_loss = []
    batchVal_loss   = []
    #### -- Set up -- ####

    #### -- Main loop -- ####
    model.to(device)
    for epoch in range(epochs):
        #tracker.epoch_start()
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

        #tracker.epoch_end()
        temp_ValLoss = np.mean(batchVal_loss)
        val_Loss.append(temp_ValLoss)

        early_stopping(temp_ValLoss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            #tracker.stop()
            break
        else:
            continue
    #if not early_stopping.early_stop:
        #tracker.stop()
    #### -- Save info -- ####
    if modelSave == True:
        t_Loss = torch.tensor(train_Loss)
        v_Loss = torch.tensor(val_Loss)
        if dev:
            torch.save(t_Loss, f = mkPathLoss + '/test' + '/train_loss' + costumLabel + model._get_name()  + now + '.pt') # add name
            torch.save(v_Loss, f = mkPathLoss + '/test' + '/val_loss' + costumLabel + model._get_name() + now + '.pt')   # add name
        else:
            torch.save(t_Loss, f = mkPathLoss + '/train_loss' + costumLabel + model._get_name()  + now + '.pt') # add name
            torch.save(v_Loss, f = mkPathLoss + '/val_loss' + costumLabel + model._get_name() + now + '.pt')   # add name

    plt.plot(range(len(train_Loss)), train_Loss, label = 'Train')
    plt.plot(range(len(val_Loss)), val_Loss, label = 'val')
    plt.axvline(val_Loss.index(min(val_Loss)), linestyle='--', color='r',label='Early Stopping Checkpoint')
    legend = plt.legend(loc = 'upper right', frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('grey')
    frame.set_edgecolor('black')
    plt.xlabel ('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.grid()
    if figSave == True:
        if dev:
            plt.savefig(mkPathLoss + '/test' +'/Figs/'+ costumLabel + model._get_name() + now + '.png', bbox_inches='tight', dpi = 400)
        else:
            plt.savefig(mkPathLoss + '/Figs/'+ costumLabel + model._get_name() + now + '.png', bbox_inches='tight', dpi = 400)
    plt.show()

    #new trying something with PRAUC
    prediction = []
    targetList = []
    if dev:
        model.load_state_dict(torch.load(mkPathLoss + '/test' + '/model' + costumLabel + model._get_name() + now))
    else:
        model.load_state_dict(torch.load(mkPathLoss + '/model' + costumLabel + model._get_name() + now))
        
    
    for batch, (data, target) in enumerate(val_Loader, 1):
        optimizer.zero_grad() # a clean up step for PyTorch
        out = model(data.to(device))
        for p in out.detach().cpu().numpy():
            prediction.append(np.argmax(p))
        targetList.append(target.detach().cpu().numpy())
        if batch % 2 == 0: #For printing
            print(4*' ', '===> F-Score: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch * len(data), len(val_Loader.dataset),
                100. * batch / len(val_Loader),
                np.mean(batchVal_loss)))
        break
    #print('Prediction shape: {0}\n{1}'.format(np.asarray(prediction).shape, prediction))
    #print('Target shape: {0}\n{1}'.format(np.asarray(targetList).shape, targetList))
    targetList = np.asarray(targetList).flatten()
    Precision, Recall, Threshold = PRAUC(prediction, targetList, ep = 1e-5)
    print('Precision: {0}\nRecall: {1}\nThreshold: {2} (not used)'.format(Precision, Recall, Threshold))

    return prediction, targetList


####### Main Calls ########

xTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainX.pt')
yTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainY.pt')

xVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValX.pt')
yVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValY.pt')

xTrain = xTrain.repeat(1, 3, 1, 1) # only for pretrained model
xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model

train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,#
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 0)

val_Set = torch.utils.data.TensorDataset(xVal, yVal)
val_Loader = torch.utils.data.DataLoader(val_Set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 0)

#param_grid = {
#    'batch_size': [16, 64],
#    'max_epochs': [10, 20]
#}
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#grid_result = grid.fit(xTrain, yTrain)
#
#print(grid_result)

p , t = TrainLoop(train_Loader = train_Loader
        , val_Loader = val_Loader
        , model    = model
        , patience = patience
        , delta    = 1e-4
        , epochs = epochs
        , optimizer = optimizer
        , loss_Fun = loss_Fun
        , modelSave = saveModel
        , figSave = figSave
        , dev = dev
        )

print('Accuracy: {0}'.format((sum([p == t]))/yVal.shape[0]))


#train = torch.load('Data/Loss_chest_xray/train_lossResNet20230309-211533.pt').tolist()
#val = torch.load('Data/Loss_chest_xray/val_lossResNet20230309-211533.pt').tolist()