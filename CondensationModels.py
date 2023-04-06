import torch
import numpy as np
import torch.nn as nn
import Models as M
import os
import warnings
import Data_processing as DP
warnings.filterwarnings("ignore")

def distance(A, B):
    upper = A* B
    lower = torch.norm(A)*torch.norm(B)
    frac = upper / lower
    return 1 - frac

def sampleRandom(data, batch_size):
    index = np.random.randint(data.shape[0], size = batch_size)
    return torch.stack([data[i] for i in index])

def gen_Y(size):
    if size % 2 != 0:
        return None
    class_0 = torch.tensor(np.asarray([0]*int(size/2)))
    class_1 = torch.tensor(np.asarray([1]*int(size/2)))
    return torch.cat((class_0, class_1))

def apply_Gradient(Gradient):
        
        return None

def fit(model, ):

    #Should return Loss
    return None

def GradientMatching(model, T_x, T_y, S_x, S_y, k, t, c, lr_Theta, lr_S, batch_size = 64):
    #S_shape = (size_of_dataset, 1, height, width) 
    #Note: assumes binary classification.
    print('init random weights...')
    model._init_weights() #Here?
    Schange_class_index = torch.argmax(S_y).item()
    Tchange_class_index = torch.argmax(T_y).item()

    # Do some setup
    for k in range(k):
        #init P0
        for t in range(t):
            for c in range(c):
                print('Generating Batches...')
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
                T_BatchX = sampleRandom(T_DataX, batch_size = batch_size)
                T_BatchY = sampleRandom(T_DataY, batch_size = batch_size)
                S_BatchX = sampleRandom(S_DataX, batch_size = batch_size)                
                S_BatchY = sampleRandom(S_DataY, batch_size = batch_size)
                print(T_BatchX.shape)
                break
                print('not implemented')
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

S_x = torch.rand((200, 3, 64, 64))
S_y = gen_Y(S_x.shape[0])
S_Set = torch.utils.data.TensorDataset(S_x, S_y)
S_Loader = torch.utils.data.DataLoader(S_Set,
                                        batch_size = 64,
                                        shuffle = False,
                                        num_workers = 0)

model = M.CD_temp()
GradientMatching(model, xTrain, yTrain, S_x, S_y, k = 1, t = 1,c = 2, lr_Theta = 1, lr_S = 1)