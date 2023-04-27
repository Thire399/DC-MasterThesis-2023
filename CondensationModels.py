import torch
import os
import warnings
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import Models as M
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics.pairwise import rbf_kernel
from carbontracker.tracker import CarbonTracker


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
        #now = time.strftime("%Y%m%d-%H%M%S")
        self.savePath = f'Data/Synthetic_{DataSet}'
        os.makedirs(f'{self.savePath}/CarbonLogs', exist_ok = True)
        self.customLabel = customLabel
        self.lr_S = lr_S
        #self.noise = torch.randn(syntheticSampleSize, 3, 128, 128)
        #self.S_x = torch.zeros(syntheticSampleSize, 3, 128, 128, dtype = torch.float)
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 3, 128, 128)), requires_grad = True) #Totally random data
        self.S_y = Gen_Y(self.S_x.shape[0])
        self.loss_Fun = loss_Fun
        self.optimizerT = optim.SGD(self.model.parameters(), lr = lr_Theta, momentum = 0.5)
        self.optimizerS = optim.SGD([self.S_x], lr = lr_S, momentum = 0.5)
        self.carbonTracker = CarbonTracker(epochs = self.k*self.t, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 0.01
                            )
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
        self.model.train()
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
        return grad_list, loss

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
        
        #c0 = random.sample(range(0, 4032), 400)
        #self.S_y = T_y[c0]
        #self.S_x = T_x[c0]
        #self.S_x = self.S_x + self.noise
        #Schange_class_index = torch.argmax(self.S_y).item()
        torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/BeforeX.pt')
        torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/BeforeY.pt')
        #print(self.S_x.shape, self.S_y.shape)
        DistanceLst = []
        for k in range(self.k):
            print('init random weights...')
            self.model._init_weights()
            for t in range(self.t):
                #self.optimizerS.zero_grad()
                #self.optimizerT.zero_grad()
                self.carbonTracker.epoch_start()
                #old = self.S_x.clone()
                print(f'K Iteration: {k}\n\tT Iteration: {t}')
                for c in range(self.c):
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
                    print(f'\t\tSampling for class {c}... ')
                    T_BatchX = self.sampleRandom(T_DataX, batch_size = self.batch_size)
                    T_BatchY = self.sampleRandom(T_DataY, batch_size = self.batch_size)
                    S_BatchX = self.sampleRandom(S_DataX, batch_size = self.batch_size)                
                    S_BatchY = self.sampleRandom(S_DataY, batch_size = self.batch_size)
                    
                    #print(T_BatchX.shape)
                    t_grad, loss = self.GetGradient(T_BatchX, T_BatchY)
                    s_grad, loss = self.GetGradient(S_BatchX, S_BatchY)
                    D = self.Distance(t_grad, s_grad)
                    D.backward()
                    DistanceLst.append(D.detach().cpu().numpy())
                    #print('distance ', D)
                    self.optimizerS.step()
                Whole_S = torch.utils.data.TensorDataset(self.S_x, self.S_y)
                S_loader = torch.utils.data.DataLoader(Whole_S
                                                        , batch_size = batch_size
                                                        , shuffle = True
                                                        , num_workers = 0)
                tempLossLst = []
                print('Training on whole S...')
                for batch, (data, target) in enumerate(S_loader, 1):
                    self.optimizerT.zero_grad() # a clean up step for PyTorch
                    out = model(data.type(torch.float32).to(self.device))
                    out = out.flatten()
                    loss = self.loss_Fun(out, (target).type(torch.float32).to(self.device))
                    loss.backward()
                    self.optimizerT.step()
                    tempLossLst.append(loss.item())
                    if batch % 2 == 0: #For printing
                        print(4*' ', '===> Training (t): [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                batch * len(data), len(S_loader.dataset),
                                (100. * batch) / len(S_loader),
                                np.mean(tempLossLst)))
                self.carbonTracker.epoch_end()
                #self.S_x = torch.tanh(self.S_x).clone()
                #temp = torch.sum(torch.eq(old, self.S_x))
                #print(f'any change? (False = Yes!  True = No!):', temp == 9830400, f'is {temp}')
        self.carbonTracker.stop()
                
        return self.S_x, self.S_y, DistanceLst


class DistributionMatching():

    def __init__(self, model, k:int, c:int,  batchSize:int, syntheticSampleSize :int
                 ,loss_Fun, lr_S:float, DataSet:str, customLabel:str) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batchSize
        self.k = k
        self.c = c
        self.savePath = f'Data/Synthetic_{DataSet}'
        os.makedirs(f'{self.savePath}/CarbonLogs', exist_ok = True)
        self.customLabel = customLabel
        self.lr_S = lr_S
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 3, 128, 128)), requires_grad = True) #Totally random data
        self.S_y = Gen_Y(self.S_x.shape[0])
        self.loss_Fun = loss_Fun
        self.optimizer = optim.SGD([self.S_x], lr = lr_S, momentum = 0.5)
        self.carbonTracker = CarbonTracker(epochs = self.k, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 0.01
                            )
        print(f'Setup:\n\tUsing Compute: {self.device}\n\tk = {k}\n \tc = {c}\n\tLearning Rate S: = {lr_S}')
    def rotate_images(images):
        rotated_images = []
        for image in images:
            # Convert tensor to PIL Image
            pil_image = TF.to_pil_image(image)
            # Rotate PIL Image
            rotated_pil_image = TF.rotate(pil_image, 15)
            # Convert rotated PIL Image back to tensor
            rotated_image = TF.to_tensor(rotated_pil_image)
            rotated_images.append(rotated_image)
        return rotated_images

    def flip_images(images):
        flipped_images = []
        for image in images:
            # Convert tensor to PIL Image
            pil_image = TF.to_pil_image(image)
            # Flip PIL Image horizontally
            flipped_pil_image = TF.hflip(pil_image)
            # Convert flipped PIL Image back to tensor
            flipped_image = TF.to_tensor(flipped_pil_image)
            flipped_images.append(flipped_image)
        return flipped_images

    def brighten_images(images, brightness_factor):
        brightened_images = []
        for image in images:
            # Convert tensor to PIL Image
            pil_image = TF.to_pil_image(image)
            # Brighten PIL Image
            brightened_pil_image = TF.adjust_brightness(pil_image, brightness_factor)
            # Convert brightened PIL Image back to tensor
            brightened_image = TF.to_tensor(brightened_pil_image)
            brightened_images.append(brightened_image)
        return brightened_images


    def Empirical_mmd(X, Y, gamma):
        K_xx = rbf_kernel(X, X, gamma)
        K_xy = rbf_kernel(X, Y, gamma)
        K_yy = rbf_kernel(Y, Y, gamma)
        mmd = np.mean(K_xx) - 2 * np.mean(K_xy) + np.mean(K_yy)
        return mmd
    
    def sampleRandom(self, data, batch_size):
        index = np.random.randint(data.shape[0], size = batch_size)
        return torch.stack([data[i] for i in index])
    def ComputeGrad(self, x):
        self.model.train()
        out = self.model(x.to(self.device))
        out = out.flatten()
        return out 
    
    def Generate(self, T_x, T_y,):
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        for k in range(self.k):
            self.model._init_weights()
            Loss_Sum = 0
            for c in range(self.c):
                print('Create Mini Batches')
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
                
                #add augmentation
                T_out = self.ComputeGrad(T_BatchX)
                S_out = self.ComputeGrad(S_BatchX)
                temp_sum_T = torch.mean(T_out)
                temp_sum_S = torch.mean(S_out) 
                loss = torch.square(torch.abs(temp_sum_T - temp_sum_S))
                Loss_Sum += loss
            Loss_Sum.backward()
            self.optimizer.step()
                #old = S_BatchX
        return self.S_x


####### PARAMETERS #######
#Data parameters
#dataSet      = 'chest_xray'
dataset = 'Alzheimer_MRI'
datatype     = ''
costumLabel  = 'Test'
andrea = True 
if andrea:
    os.chdir(r"C:\Users\andre\Documents\GitHub\DC-MasterThesis-2023")
else:
    os.chdir('/home/thire399/Documents/School/DC-MasterThesis-2023')
batch_size   = 64
####### PARAMETERS #######
print('preparing training data...')
#Train data
xTrain = torch.load(f'Data/Proccesed/{dataset}/trainX.pt')
yTrain = torch.load(f'Data/Proccesed/{dataset}/trainY.pt')
xTrain = xTrain.repeat(1, 3, 1, 1)

train_Set = torch.utils.data.TensorDataset(xTrain, yTrain)
train_Loader = torch.utils.data.DataLoader(train_Set,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 0)


model = M.ConvNet()#M.CD_temp()
print('\nStaring Condensation...\n')
#GM = GradientMatching(model
#                        , batchSize = 64
#                        , syntheticSampleSize = 400
#                        , k = 1000
#                        , t = 50
#                        , c = 2
#                        , lr_Theta = 0.01
#                        , lr_S = 0.1
#                        , loss_Fun = nn.BCEWithLogitsLoss()
#                        , DataSet = dataset
#                        , customLabel = costumLabel)

DM = DistributionMatching(model
                        , batchSize = 8
                        , syntheticSampleSize = 200
                        , k = 200
                        , c = 2
                        #, lr_Theta = 0.01
                        , lr_S = 1e-3
                        , loss_Fun = nn.BCEWithLogitsLoss()
                        , DataSet = dataset
                        , customLabel = costumLabel)

#x, y, d = GM.Generate(xTrain, yTrain)
#GM.save_output()

x = DM.Generate(xTrain, yTrain)
#DM.save_output()

x = x.cpu().detach().numpy()
#plt.plot(range(len(d)), d)
#plt.show()


#print(y[0])
#plt.imshow(x[0][0], cmap = 'gray')
##plt.savefig('Data/Loss_chest_xray/test/Test.png', dpi = 400, bbox_inches = 'tight')
#plt.show()
