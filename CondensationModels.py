import torch
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import Models as M
import torch.nn as nn
import torch.optim as optim
import Data_processing as DP
warnings.filterwarnings("ignore")

def distance(A, B):
    _sum_ = 0
    for i in range(len(A)):
        upper = torch.dot(A[i], B[i])
        lower = torch.norm(A[i]) * torch.norm(B[i])
        frac = upper / lower
        _sum_ += 1 - frac
    return torch.tensor(_sum_, requires_grad = True)

def gen_dataset(x, y):
    Set = torch.utils.data.TensorDataset(x, y)
    train_Loader = torch.utils.data.DataLoader(Set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 0)
    return train_Loader

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

def GetGradient(model, x, y, loss_Fun):
    model.train()
    out = model(x)
    loss = loss_Fun(out, y)
    loss.backward()
    grad_list = []
    for i, l in enumerate(model.modules()):
        if i == 0:
            pass
        else:
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                #print(d, d.weight.grad.flatten())
                grad_list.append(l.weight.grad.flatten())
    return grad_list, loss

def GradientMatching(model, T_x, T_y, S_x, S_y, k, t, c, lr_Theta, lr_S, batch_size = 64):
    loss_Fun     = nn.CrossEntropyLoss()
    #S_shape = (size_of_dataset, 1, height, width) 
    #Note: assumes binary classification.
    Schange_class_index = torch.argmax(S_y).item()
    Tchange_class_index = torch.argmax(T_y).item()
    #optim.SGD([{'params': model.parameters()}, {'params': S_x, 'lr': lr_S}], lr = lr_Theta)
    optimizerT = optim.SGD(model.parameters(), lr = lr_Theta)
    optimizerS = optim.SGD([{'params':[S_x], 'lr': lr_S}])
    # Do some setup
    for k in range(k):
        print('init random weights...')
        model._init_weights() #Here?
        for t in range(t):
            #if t % 2 == 0:
            print(f'K Iteration: {k}\n\tT Iteration: {t}')
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
                old = S_BatchX
                #print(T_BatchX.shape)
                t_grad, loss = GetGradient(model, T_BatchX, T_BatchY, loss_Fun)
                s_grad, loss = GetGradient(model, S_BatchX, S_BatchY, loss_Fun)
                D = distance(t_grad, s_grad)
                D.backward()
                print('distance ', D)
                optimizerS.step()
#                print(S_BatchX.shape, type(S_BatchX))
#                print(old.shape, type(old))
                #print(torch.sum(torch.eq(S_BatchX[0], old[0])))
#                print(torch.all(torch.eq(S_BatchX[0], old[0])))
#                    print(True)
                #print('not Finished implemented')
        
            optimizerT.step()
            print('loss per iter', loss.item())
    return S_x, S_y


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

#train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
#train_Loader = torch.utils.data.DataLoader(train_Set,
#                                        batch_size = batch_size,
#                                        shuffle = True,
#                                        num_workers = 0)

S_x = torch.rand((200, 3, 64, 64))
S_y = gen_Y(S_x.shape[0])
#S_Set = torch.utils.data.TensorDataset(S_x, S_y)
#S_Loader = torch.utils.data.DataLoader(S_Set,
#                                        batch_size = 64,
#                                        shuffle = False,
#                                        num_workers = 0)

model = M.CD_temp()
print('\nStaring Condensation...\n')
x, y = GradientMatching(model, xTrain, yTrain, S_x, S_y, k = 10, t = 10,c = 2, lr_Theta = 1, lr_S = 1)

print(y[0])
plt.imshow(S_x[0][0])
plt.savefig('Data/Loss_chest_xray/test/Test.png', dpi = 400, bbox_inches = 'tight')
plt.show()

'''
import Models as M
import torch.nn as nn
import torch

rnd_img = torch.rand((1, 3, 64, 64), requires_grad=True)
loss_Fun     = nn.CrossEntropyLoss()
temp = M.CD_temp()

temp._init_weights()
temp.train()
temp(rnd_img)
for e in range(1):
    out = temp(rnd_img)
    loss = loss_Fun(out, (torch.tensor([1])).type(torch.LongTensor))
    loss.backward()
    for i, d in enumerate(temp.modules()):
        if i == 0:
            pass
        else:
            if isinstance(d, nn.Linear) or isinstance(d, nn.Conv2d):
                #print(d, d.weight.grad.flatten())
                grad = d.weight.grad.flatten()

def distance(A, B):
    upper = A * B
    lower = torch.norm(A)*torch.norm(B)
    frac = upper / lower
    return 1 - frac

print(torch.norm(grad))
    
import torch, torch.nn as nn
x = nn.Linear(100, 100)
nn.init.normal_(x.weight, mean=0, std=1.0)
            

'''