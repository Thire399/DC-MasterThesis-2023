import torch
import os
import warnings
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import Models as M
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import rbf_kernel
from carbontracker.tracker import CarbonTracker
import gc


warnings.filterwarnings("ignore")

def Gen_Y(size):
    if size % 2 != 0:
        return None
    class_0 = torch.tensor(np.asarray([0]*int(size/2)))
    class_1 = torch.tensor(np.asarray([1]*int(size/2)))
    return torch.cat((class_0, class_1))

class GradientMatching():
    def __init__(self, model, batchSize:int, syntheticSampleSize:int,
                  k:int, t:int, c:int, lr_Theta:float, lr_S:float
                  , loss_Fun, DataSet:str, customLabel: str) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batchSize
        self.k   = k
        self.t   = t
        self.c   = c
        self.lr_Theta = lr_Theta
        self.savePath = f'Data/Synthetic_{DataSet}'
        os.makedirs(f'{self.savePath}/CarbonLogs', exist_ok = True)
        self.customLabel = customLabel
        self.lr_S = lr_S
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 1, 128, 128)), requires_grad = True) #Totally random data
        self.S_y = Gen_Y(self.S_x.shape[0])
        self.loss_Fun = loss_Fun
        self.optimizerT = optim.SGD(self.model.parameters(), lr = lr_Theta, momentum = 0.5)
        self.optimizerS = optim.SGD([self.S_x], lr = lr_S, momentum = 0.5)
        self.carbonTracker = CarbonTracker(epochs = self.k, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + self.model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 1
                            )
        self.sigmoid = nn.Sigmoid()
        print(f'Setup:\n\tUsing Compute: {self.device}\n\tk = {k}\n\tt = {t}\n\tc = {c}\n\tLearning Rate S: = {lr_S}',
                            f'\tLearning Rate Theta = {lr_Theta}')

        
    def Distance(self, A, B):
        _sum_ = 0
        for i in range(len(A)):
            upper = torch.dot(A[i], B[i])
            lower = torch.norm(A[i]) * torch.norm(B[i])
            frac = upper / lower
            _sum_ += 1 - frac
        return torch.tensor(_sum_, requires_grad = True)
    
    def sampleRandom(self, data, batch_size):
        index = np.random.randint(data.shape[0], size = batch_size)
        return torch.stack([data[i] for i in index])

    def GetGradient(self, x, y):
        self.model.eval()
        out = self.model(x.to(self.device))
        out = out.flatten()
        loss = self.loss_Fun(out, y.type(torch.float32).to(self.device))
        loss.backward()
        grad_list = []
        for i, l in enumerate(self.model.modules()):
            if i == 0:
                pass
            else:
                if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                    grad_list.append(l.weight.grad.flatten())
        del loss
        gc.collect() # cleanup step. (Loss here not used)
        return grad_list

    def save_output(self, x = None, y = None) -> None:
        print('Saving synthetic dataset...')
        if x == None:
            torch.save(self.S_x, f = f'{self.savePath}/{self.customLabel}X.pt')
            torch.save(self.S_y, f = f'{self.savePath}/{self.customLabel}Y.pt')
        else:
            torch.save(x, f = f'{self.savePath}/{self.customLabel}X.pt')
            torch.save(y, f = f'{self.savePath}/{self.customLabel}Y.pt')
        print(f'Saved "{self.customLabel}" to "{self.savePath}"')
        return None

    def Generate(self, T_x, T_y):
        #Note: assumes binary classification.
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        
        torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/GMBeforeX.pt')
        torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/GMBeforeY.pt')
        DistanceLst = []

        for k in range(self.k):
            self.carbonTracker.epoch_start()
            print('init random weights...')
            self.model._init_weights()
            for t in range(self.t):
                self.optimizerS.zero_grad()
                self.optimizerT.zero_grad()
                #old = self.S_x.clone()
                if t % 5 == 0:
                    printout = True
                    print(f'K Iteration: {k}\n\tT Iteration: {t}')
                else: printout = False

                for c in range(self.c):
                    if printout:
                        print('\t\tGenerating Batches...')
                    if c == 0:
                        T_DataX = torch.tensor(T_x[:Tchange_class_index])
                        T_DataY = torch.tensor(T_y[:Tchange_class_index])
                        S_DataX = torch.tensor(self.S_x[:Schange_class_index])
                        S_DataY = torch.tensor(self.S_y[:Schange_class_index])
                    else:
                        T_DataX = torch.tensor(T_x[Tchange_class_index:])
                        T_DataY = torch.tensor(T_y[Tchange_class_index:])
                        S_DataX = torch.tensor(self.S_x[Schange_class_index:])
                        S_DataY = torch.tensor(self.S_y[Schange_class_index:])
                    if printout:
                        print(f'\t\tSampling for class {c}... ')
                    T_BatchX = self.sampleRandom(T_DataX, batch_size = self.batch_size)
                    T_BatchY = self.sampleRandom(T_DataY, batch_size = self.batch_size)
                    S_BatchX = self.sampleRandom(S_DataX, batch_size = self.batch_size)                
                    S_BatchY = self.sampleRandom(S_DataY, batch_size = self.batch_size)
                    del T_DataX
                    del T_DataY
                    del S_DataX
                    del S_DataY
                    gc.collect() # clean up
                    t_grad = self.GetGradient(T_BatchX, T_BatchY)
                    s_grad = self.GetGradient(S_BatchX, S_BatchY)
                    del T_BatchX
                    del T_BatchY
                    del S_BatchY
                    del S_BatchX
                    gc.collect() # clean up
                    D = self.Distance(t_grad, s_grad)
                    D.backward()
                    DistanceLst.append(D.detach().cpu().numpy())
                    self.optimizerS.step()
                    del t_grad
                    del s_grad
                    gc.collect() # clean up
                Whole_S = torch.utils.data.TensorDataset(self.S_x, self.S_y)
                S_loader = torch.utils.data.DataLoader(Whole_S
                                                        , batch_size = batch_size
                                                        , shuffle = True
                                                        , num_workers = 0)
                tempLossLst = []
                if printout:
                    print('Training on whole S...')
                self.model.train()
                # Training on whole S
                for batch, (data, target) in enumerate(S_loader, 1):
                    self.optimizerT.zero_grad() # a clean up step for PyTorch
                    out = self.model(data.type(torch.float32).to(self.device))
                    out = out.flatten()
                    loss = self.loss_Fun(out, (target).type(torch.float32).to(self.device))
                    loss.backward(retain_graph = True)
                    self.optimizerT.step()
                    tempLossLst.append(loss.item())
                    if printout:
                        if batch % 2 == 0: #For printing
                            print(4*' ', '===> Training (t): [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    batch * len(data), len(S_loader.dataset),
                                    (100. * batch) / len(S_loader),
                                    np.mean(tempLossLst)))
                

                self.S_x = self.sigmoid(self.S_x) #Replace tanh -> sigmoid? [0-1]

                torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/GMIntermidiateX.pt')
                torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/GMIntermidiateY.pt')
            self.carbonTracker.epoch_end()

        self.carbonTracker.stop()
                
        return self.S_x, self.S_y, DistanceLst


class DistributionMatching():

    def __init__(self, model, k:int, c:int,  batchSize:int, syntheticSampleSize :int
                 ,loss_Fun, lr_S:float, lr_Theta:float , DataSet:str, customLabel:str) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batchSize
        self.k = k
        self.c = c
        self.savePath = f'Data/Synthetic_{DataSet}'
        os.makedirs(f'{self.savePath}/CarbonLogs', exist_ok = True)
        self.customLabel = customLabel
        self.lr_S = lr_S
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 1, 128, 128)), requires_grad = True) #Totally random data
        self.S_y = Gen_Y(self.S_x.shape[0])
        self.loss_Fun = loss_Fun
        self.optimizerT = optim.SGD([self.S_x], lr = lr_Theta, momentum = 0.5)
        self.optimizerS = optim.SGD([self.S_x], lr = lr_S, momentum = 0.5)
        self.carbonTracker = CarbonTracker(epochs = self.k, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 0.01
                            )
        print(f'Setup:\n\tUsing Compute: {self.device}\n\tk = {k}\n \tc = {c}\n\tLearning Rate S: = {lr_S}',
                            f'\tLearning Rate Theta = {lr_Theta}')

    def Empirical_mmd(X, Y, gamma):
        K_xx = rbf_kernel(X, X, gamma)
        K_xy = rbf_kernel(X, Y, gamma)
        K_yy = rbf_kernel(Y, Y, gamma)
        mmd = np.mean(K_xx) - 2 * np.mean(K_xy) + np.mean(K_yy)
        return mmd
    
    def sampleRandom(self, data, batch_size):
        index = np.random.randint(data.shape[0], size = batch_size)
        return torch.stack([data[i] for i in index])
    

    def save_output(self, x = None, y = None) -> None:
        print('Saving synthetic dataset...')
        if x == None:
            torch.save(self.S_x, f = f'{self.savePath}/{self.customLabel}X.pt')
            torch.save(self.S_y, f = f'{self.savePath}/{self.customLabel}Y.pt')
        else:
            torch.save(x, f = f'{self.savePath}/{self.customLabel}X.pt')
            torch.save(y, f = f'{self.savePath}/{self.customLabel}Y.pt')
        print(f'Saved "{self.customLabel}" to "{self.savePath}"')
        return None

    def Generate(self, T_x, T_y,):
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/DMBeforeX.pt')
        torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/DMBeforeY.pt')
        embed = self.model.module.avgpool if torch.cuda.device_count() > 1 else self.model.avgpool # for GPU parallel
        self.model.to(self.device)
        for k in range(self.k):
            #Sample paratameters for network. 
            self.model._init_weights()
            Loss_Sum = 0 #??? skal den være her? 
            self.optimizerS.zero_grad()
            self.optimizerT.zero_grad
            self.carbonTracker.epoch_start()
            if k % 5 == 0:
                printout = True
                print(f'K Iteration: {k}')
            else: printout = False
            for c in range(self.c):
                print('Create Mini Batches')
                if c == 0:
                    T_DataX = (T_x[:Tchange_class_index])
                    T_DataY = (T_y[:Tchange_class_index])
                    S_DataX = (self.S_x[:Schange_class_index])
                    S_DataY = (self.S_y[:Schange_class_index])
                    #sample w_c - omega for every class
                else:
                    T_DataX = (T_x[Tchange_class_index:])
                    T_DataY = (T_y[Tchange_class_index:])
                    S_DataX = (self.S_x[Schange_class_index:])
                    S_DataY = (self.S_y[Schange_class_index:])
                    #sample w_c - omega for every class
                if printout:
                        print(f'\t\tSampling for class {c}... ')
                print('Sampling...')
                T_BatchX = self.sampleRandom(T_DataX, batch_size = self.batch_size)
                T_BatchY = self.sampleRandom(T_DataY, batch_size = self.batch_size)
                S_BatchX = self.sampleRandom(S_DataX, batch_size = self.batch_size)                
                S_BatchY = self.sampleRandom(S_DataY, batch_size = self.batch_size)
                
                #compute MDD and add augmentation
                #T_out = self.Compute(T_BatchX)
                #S_out = self.Compute(S_BatchX)
                #print(T_BatchX[1][1])
                #print(T_BatchX)

                T_aug = TF.rotate(T_BatchX, 0.15)
                S_aug = TF.rotate(S_BatchX, 0.15)
                # get embeddings
                T_embed = embed(T_aug)
                S_embed = embed(S_aug)


                # compute the loss
                loss = torch.sum((torch.mean(T_embed, dim=0) - torch.mean(S_embed, dim=0))**2)
                Loss_Sum += loss
                self.optimizerS.step()

            self.S_x = nn.Sigmoid()(self.S_x)
            self.carbonTracker.epoch_end()
            # backpropagation and weight update

            Loss_Sum.backward()
            self.optimizerT.step()            
            torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/DMIntermidiateX.pt')
            torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/DMIntermidiateY.pt')
        self.carbonTracker.stop()
        return self.S_x, self.S_y


####### PARAMETERS #######
#Data parameters
#dataSet      = 'chest_xray'
dataset = 'Alzheimer_MRI'
datatype     = ''

costumLabel  = 'GMAfter'
andrea = False
server = True
if server:
    os.chdir("/home/datacond/Documents/school/To_Server")
elif andrea:
    os.chdir('/Users/andreamoody/Documents/GitHub/DC-MasterThesis-2023')
else:
    os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')
batch_size   = 32
####### PARAMETERS #######
print('preparing training data...')
#Train data
xTrain = torch.load(f'Data/Proccesed/{dataset}/trainX.pt')
yTrain = torch.load(f'Data/Proccesed/{dataset}/trainY.pt')
#xTrain = xTrain.repeat(1, 3, 1, 1)

train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 4)

print('\nStaring Condensation...\n')
model = M.ConvNet()
GM = GradientMatching(model
                        , batchSize = 64
                        , syntheticSampleSize = 100
                        , k = 10
                        , t = 50
                        , c = 2
                        , lr_Theta = 0.01
                        , lr_S = 0.1
                        , loss_Fun = nn.BCEWithLogitsLoss()
                        , DataSet = dataset
                        , customLabel = costumLabel)
model = M.ConvNet2(output_layer='avgpool')#M.CD_temp()
#DM = DistributionMatching(model
#                        , batchSize = 32
#                        , syntheticSampleSize = 100
#                        , k = 10
#                        , c = 2
#                        , lr_Theta = 0.01
#                        , lr_S = 1
#                        , loss_Fun = nn.BCEWithLogitsLoss()
#                        , DataSet = dataset
#                        , customLabel = costumLabel)

x, y, d = GM.Generate(xTrain, yTrain)
GM.save_output()

#x = DM.Generate(xTrain, yTrain)
#DM.save_output()

#x = x.cpu().detach().numpy()
#plt.plot(range(len(d)), d)
#plt.show()


#print(y[0])
#plt.imshow(x[0][0], cmap = 'gray')
##plt.savefig('Data/Loss_chest_xray/test/Test.png', dpi = 400, bbox_inches = 'tight')
#plt.show()
