import torch
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import Models as M
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import rbf_kernel
import Data_processing as DP
warnings.filterwarnings("ignore")

def Gen_Y(size):
    if size % 2 != 0:
        return None
    class_0 = torch.tensor(np.asarray([0]*int(size/2)))
    class_1 = torch.tensor(np.asarray([1]*int(size/2)))
    return torch.cat((class_0, class_1))

class GradientMatching():
    def __init__(self, model, batchSize, syntheticSampleSize, k, t, c, lr_Theta, lr_S, loss_Fun) -> None:
        self.model = model
        self.batch_size = batchSize
        self.k   = k
        self.t   = t
        self.c   = c
        self.lr_Theta = lr_Theta
        self.lr_S = lr_S
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 3, 64, 64)))
        self.S_y = Gen_Y(self.S_x.shape[0])
        self.loss_Fun = loss_Fun #self.loss_Fun     = nn.CrossEntropyLoss()
        #self.synthetic = torch.utils.data.TensorDataset(S_x, S_y)
        self.optimizerT = optim.SGD(self.model.parameters(), lr = lr_Theta)
        self.optimizerS = optim.SGD([{'params':[self.S_x], 'lr': lr_S}])
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
        self.model.train()
        out = self.model(x)
        loss = self.loss_Fun(out, y)
        loss.backward()
        grad_list = []
        for i, l in enumerate(self.model.modules()):
            if i == 0:
                pass
            else:
                if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                    grad_list.append(l.weight.grad.flatten())
        return grad_list, loss

    def Generate(self, T_x, T_y):
        #S_shape = (size_of_dataset, 1, height, width) 
        #Note: assumes binary classification.
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        # Do some setup
        for k in range(self.k):
            print('init random weights...')
            self.model._init_weights() #Here?
            for t in range(self.t):
                old = self.S_x.clone()
                #if t % 2 == 0:
                print(f'K Iteration: {k}\n\tT Iteration: {t}')
                for c in range(self.c):
                    print('Generating Batches...')
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
                    print('Sampling...')
                    T_BatchX = self.sampleRandom(T_DataX, batch_size = batch_size)
                    T_BatchY = self.sampleRandom(T_DataY, batch_size = batch_size)
                    S_BatchX = self.sampleRandom(S_DataX, batch_size = batch_size)                
                    S_BatchY = self.sampleRandom(S_DataY, batch_size = batch_size)
                    
                    #print(T_BatchX.shape)
                    t_grad, loss = self.GetGradient(T_BatchX, T_BatchY)
                    s_grad, loss = self.GetGradient(S_BatchX, S_BatchY)
                    D = self.Distance(t_grad, s_grad)
                    D.backward()
                    print('distance ', D)
                    self.optimizerS.step()
                self.optimizerT.step()
                print('any change?:', torch.sum(torch.eq(old, self.S_x)) == 2457600)
                print(self.S_x.requires_grad)
                print('loss per iter', loss.item())
        return self.S_x, self.S_y


class DistributionMatching():
    def __init__(self, model, k = 200, c=2,  batchSize = 64, syntheticSampleSize = 200):
        self.model = model
        self.k = k
        self.c = c
        self.batch_size = batchSize
        S_x = torch.rand((syntheticSampleSize, 3, 64, 64))
        S_y = Gen_Y(S_x.shape[0])
        self.synthetic = torch.utils.data.TensorDataset(S_x, S_y)

    def Empirical_mmd(X, Y, gamma):
        K_xx = rbf_kernel(X, X, gamma)
        K_xy = rbf_kernel(X, Y, gamma)
        K_yy = rbf_kernel(Y, Y, gamma)
        mmd = np.mean(K_xx) - 2 * np.mean(K_xy) + np.mean(K_yy)
        return mmd

    def sampleRandom(self, data, batch_size):
        index = np.random.randint(data.shape[0], size = batch_size)
        return torch.stack([data[i] for i in index])

    def DM(self, T_x, T_y, S_x, S_y,):
        Schange_class_index = torch.argmax(S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        for k in range(self.k):
            for c in range(self.c):
                print('Create Mini Beatches')
                if c == 0:
                    T_DataX = torch.tensor(T_x[:Tchange_class_index])
                    T_DataY = torch.tensor(T_y[:Tchange_class_index])
                    S_DataX = torch.tensor(S_x[:Schange_class_index])
                    S_DataY = torch.tensor(S_y[:Schange_class_index])
                    
                else:
                    T_DataX = torch.tensor(T_x[Tchange_class_index:])
                    T_DataY = torch.tensor(T_y[Tchange_class_index:])
                    S_DataX = torch.tensor(S_x[Schange_class_index:])
                    S_DataY = torch.tensor(S_y[Schange_class_index:])
                print('Sampling...')
                T_BatchX = self.sampleRandom(T_DataX, batch_size = batch_size)
                T_BatchY = self.sampleRandom(T_DataY, batch_size = batch_size)
                S_BatchX = self.sampleRandom(S_DataX, batch_size = batch_size)                
                S_BatchY = self.sampleRandom(S_DataY, batch_size = batch_size)
                old = S_BatchX

    
        return None


####### PARAMETERS #######
#Data parameters
dataSet      = 'chest_xray'
datatype     = ''
costumLabel  = '64x64Full'
os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')
batch_size   = 64
####### PARAMETERS #######
print('preparing training data...')
#Train data

#Concat the two types
xTrain, yTrain = DP.DataPrep('Data/Proccesed/chest_xray/train/trainnormal.pt', 'Data/Proccesed/chest_xray/train/trainpneumonia.pt')

xTrain = xTrain.repeat(1, 3, 1, 1)

train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 0)


model = M.CD_temp()
print('\nStaring Condensation...\n')
GM = GradientMatching(model
                        , batchSize = 64
                        , syntheticSampleSize = 200
                        ,k = 10
                        , t = 10
                        ,c = 2
                        , lr_Theta = 1
                        , lr_S = 1
                        , loss_Fun = nn.CrossEntropyLoss())
x, y = GM.Generate(xTrain, yTrain)
#
#print(y[0])
#plt.imshow(S_x[0][0])
#plt.savefig('Data/Loss_chest_xray/test/Test.png', dpi = 400, bbox_inches = 'tight')
#plt.show()
