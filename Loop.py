import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import Models as M
import re
from torchvision import models
from carbontracker.tracker import CarbonTracker
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score



# Maybe move to another file?
def PRAUC(pred, Target, ep):
    '''
    Takes in output from model and Target labels,
    and an epsilong to avoid div by 0
    Calculates the precision and recall for the given model over multiple 
    thressholds, by using the sklearn implementation
    https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
    '''

    P, R, T = precision_recall_curve(y_true = Target, probas_pred = pred)
    fscore = (2 * P * R) / (P + R + ep)

    # locate the index of the largest f score
    ix = np.argmax(fscore)
    bestT = T[ix]
    print('Best Threshold=%f, F-Score=%.3f' % (T[ix], fscore[ix]))
    Prediction = [int(round(x[0])) if x[0] >= bestT else 0 for x in pred]
    return P[ix], R[ix], Prediction, fscore[ix], bestT

def TrainLoop(train_Loader, val_Loader, model, patience, delta, epochs, optimizer, loss_Fun, modelSave, figSave, dataSet, costumLabel, dev = False, isBinary = True):
    
    now = time.strftime("%Y%m%d-%H%M%S") #save file as current time stamp - better format to save file?
    mkPathLoss = 'Data/Loss_' + dataSet
    if dev:
        print()
#        os.makedirs(f'Data/Loss_{dataSet}/test', exist_ok = True)
        tempPath = mkPathLoss + '/test/CarbonLogs'
        os.makedirs(tempPath, exist_ok= True)
    else:
        tempPath = mkPathLoss + '/CarbonLogs'
        os.makedirs(tempPath, exist_ok= True)
    tracker = CarbonTracker(epochs=epochs, 
                            log_dir = tempPath,
                            log_file_prefix = costumLabel + model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 0.01
                            )

    #### -- Set up -- ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    if device == 'cpu':
         print('Not gonna run on the CPU')
         return None


    os.makedirs(mkPathLoss , exist_ok = True)
    os.makedirs(mkPathLoss + '/Figs' , exist_ok = True)

    if dev:
        early_path = mkPathLoss + '/test/model' + costumLabel + model._get_name() + now

    else:
        early_path = mkPathLoss + '/model' + costumLabel + model._get_name() + now
    early_stopping = M.EarlyStopping( patience = patience,
                    verbose = True,
                    delta   = delta,
                    path    =  early_path,
                    saveModel = modelSave)
        
    train_Loss = []
    val_Loss = []
    batchTrain_loss = []
    batchVal_loss   = []
    #### -- Set up -- ####
    torch.autograd.detect_anomaly()
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
            out = model(data.type(torch.float32).to(device))
            if isBinary:
                out = out.flatten()
                target = target.type(torch.float32)
            else:
                #out = torch.softmax(out, dim=1)
                # Getting the index of the maximum probability (predicted class)
                target = target.type(torch.int64)
            loss = loss_Fun(out, (target).to(device))
            loss.backward()
            optimizer.step()
            batchTrain_loss.append(loss.item())
            if batch % 8 == 0:
                print('Train Epoch [{}/{}]: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, epochs, batch * len(data), len(train_Loader.dataset),
                    100. * batch / len(train_Loader),
                    np.mean(batchTrain_loss)))
            
        train_Loss.append(np.mean(batchTrain_loss))
        #time.sleep(2) #used when loop too fast for carbon trackerf
        # Val Data.
        model.eval()
        for batch, (data, target) in enumerate(val_Loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            out = model(data.type(torch.float32).to(device))
            if isBinary:
                out = out.flatten()
                target = target.type(torch.float32)
            else:
                #out = torch.softmax(out, dim=1)
                target = target.type(torch.int64)
                # Getting the index of the maximum probability (predicted class)
                #out = torch.argmax(probs, dim=1)
            #out = out.clamp(min = 0)
            loss = loss_Fun(out, (target).to(device))
            batchVal_loss.append(loss.item())
            if batch % 8 == 0: #For printing
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
            tracker.stop()
            break
        else:
            continue
        
    if not early_stopping.early_stop:
        tracker.stop()
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
    plt.title(f'{model._get_name()} {costumLabel} Loss per epoch')
    plt.grid()
    if figSave == True:
        if dev:
            os.makedirs(mkPathLoss + '/test' +'/Figs/', exist_ok = True)
            plt.savefig(mkPathLoss + '/test' +'/Figs/'+ costumLabel + model._get_name() + now + '.png', bbox_inches='tight', dpi = 400)
        else:
            plt.savefig(mkPathLoss + '/Figs/'+ costumLabel + model._get_name() + now + '.png', bbox_inches='tight', dpi = 400)
    plt.show()
    plt.close()

    return None

def eval_model(model, dataset, dev, val_Loader,  model_filePath = None, size = '64x64', threshold = None, isPrint = True, isbinary = True):
    torch.cuda.empty_cache()
        #new trying something with PRAUC
    predictionList = []
    targetList = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    if device == 'cpu':
         print('Not gonna run on the CPU')
         return None
    if dev:
        if model_filePath == None:
            model_filePath = np.sort(np.asarray([os.path.join('Data/Loss_' + dataset + '/test', file) for file 
                                                 in os.listdir('Data/Loss_' + dataset + '/test') 
                                                 if re.search('model', file) and re.search(size+model._get_name(), file) ]))[-1]
            print(f'no model specified.\nUsing last trained model: "{model_filePath}"')
        else:
            print(f'Model specified.\nUsing trained model: "{model_filePath}"') 
        model.load_state_dict(torch.load(model_filePath))
    else:
        if model_filePath == None:
            model_filePath = np.sort(np.asarray([os.path.join('Data/Loss_' + dataset, file) for file 
                                                 in os.listdir('Data/Loss_' + dataset)
                                                   if re.search('model', file) and re.search(size+model._get_name(), file)]))[-1]  
            print(f'no model specified.\nUsing last trained model: "{model_filePath}"')  
        else:
            print(f'Model specified.\nUsing trained model: "{model_filePath}"')
        model.load_state_dict(torch.load(model_filePath))
    print(model_filePath)
    model.to(device)
    model.eval()
    for batch, (data, target) in enumerate(val_Loader, 1):
        out = model(data.type(torch.float32).to(device))
        if isbinary: #using sigmoid to go from logits to probabilities.
            out = torch.sigmoid(out)
        else: out = torch.softmax(out, dim = 1)
        #for each batch get each prediction out + the target.
        for p in out.detach().cpu().numpy():
            predictionList.append(p) #np.argmax(p))
        for t in target.detach().cpu().numpy():
            targetList.append(t)
        if batch % 8 == 0 and isPrint: #For printing
            print(4*' ', '===> F-Score: [{}/{} ({:.0f}%)]\t'.format(
                batch * len(data), len(val_Loader.dataset),
                100. * batch / len(val_Loader)))    
    
    targetList = np.array(targetList).flatten()
    if threshold == None:
        if isbinary:
            Precision, Recall, Prediction, fscore, threshold = PRAUC(predictionList, targetList, ep = 1e-5)
            print('Precision: {0}\nRecall: {1}'.format(Precision, Recall))
        else:
            _, counts = np.unique(targetList, return_counts = True)
            if len(np.unique(counts)) > 1: #if more than 1 unique counts = class imbalance
                mode = 'weighted' # for class imbalance
            else: mode = 'macro'
            predictionList = np.argmax(np.array(predictionList), axis = 1)
            Prediction = precision_score(targetList, predictionList, average = mode)
            recall = recall_score(targetList, predictionList, average = mode)
            fscore = f1_score(targetList, predictionList, average = mode)
            threshold = 0.5 #not used for multiclass.
            print(f'Precision: {Prediction}\nRecall: {recall}\nfscore: {fscore}')

    else:
        print(threshold)
        Prediction = [int(round(x[0])) if x[0] >= threshold else 0 for x in predictionList]
        fscore = f1_score(targetList, Prediction, average = 'binary')
        print('F1 score:{0}'.format(fscore))
    return fscore, Prediction, threshold#, predictionList


####### Main Calls ########

#xTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainX.pt')
#yTrain = torch.load('Data/Proccesed/'+ dataSet +'/' + datatype + 'trainY.pt')
#
#xVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValX.pt')
#yVal = torch.load('Data/Proccesed/'+ dataSet + '/' + datatype +'tempValY.pt')
#
#xTrain = xTrain.repeat(1, 3, 1, 1) # only for pretrained model
#xVal = xVal.repeat(1, 3, 1, 1)     # only for pretrained model
#
#train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
#train_Loader = torch.utils.data.DataLoader(train_Set,
#                                            batch_size = batch_size,
#                                            shuffle = True,
#                                            num_workers = 0)
#
#val_Set = torch.utils.data.TensorDataset(xVal, yVal)
#val_Loader = torch.utils.data.DataLoader(val_Set,
#                                            batch_size = batch_size,
#                                            shuffle = True,
#                                            num_workers = 0)

#param_grid = {
#    'batch_size': [16, 64],
#    'max_epochs': [10, 20]
#}
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#grid_result = grid.fit(xTrain, yTrain)
#
#print(grid_result)

# p = prediction, t = target
#TrainLoop(train_Loader = train_Loader
#        , val_Loader = val_Loader
#        , model    = model
#        , patience = patience
#        , delta    = 1e-4
#        , epochs = epochs
#        , optimizer = optimizer
#        , loss_Fun = loss_Fun
#        , modelSave = saveModel
#        , figSave = figSave
#        , dev = dev
#        )

#p, t = eval_model(model = model
#                  , dataset = dataSet
#                  , dev = dev
#                  , val_Loader = val_Loader)
#print('Accuracy on temp ValidationSet: {0}     --> (sum(Prediction = Target))/n_sampels'.format(np.sum([p == t])/t.shape[0]))