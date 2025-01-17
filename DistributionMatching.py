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
# FOR PRINTINT #
# ANSI escape codes for different colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
BOLD = "\u001b[1m"
RESET = '\033[0m'
########################

warnings.filterwarnings("ignore")

def Gen_Y(size):
    if size % 2 != 0:
        return None
    class_0 = torch.tensor(np.asarray([0]*int(size/2)))
    class_1 = torch.tensor(np.asarray([1]*int(size/2)))
    return torch.cat((class_0, class_1))

class DistributionMatching():

    def __init__(self, model, k:int, c:int,  batchSize:int, syntheticSampleSize :int
                 ,loss_Fun, lr_S:float , DataSet:str, customLabel:str) -> None:
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
        self.optimizerS = optim.SGD([self.S_x], lr = lr_S, momentum = 0.5)
        self.carbonTracker = CarbonTracker(epochs = 1, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 0.01
                            )
        print(f'Setup:\n\tUsing Compute: {self.device}\n\tk = {k}\n \tc = {c}\n\tLearning Rate S: = {lr_S}')
    
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
    
    def aug_strategy(self, true_x, syn_x):
        rand_nr = np.random.randint(1,3)
        if rand_nr == 1:
            T_aug = TF.rotate(true_x, 0.15)
            S_aug = TF.rotate(syn_x, 0.15)
        elif rand_nr == 2:
            T_aug = torch.flip(true_x, [2])
            S_aug = torch.flip(syn_x, [2])
        else: 
            T_aug = TF.adjust_brightness(true_x, 0,2)
            S_aug = TF.adjust_brightness(syn_x, 0,2)
        return T_aug, S_aug
    
    def sigmoid(self, x):
        """Sigmoid Activation Function
            Arguments:
            x.torch.tensor
            Returns
            Sigmoid(x.torch.tensor)
        """
        return 1 / (1+torch.exp(-x))
    def min_max_normalization(self, images_tensor): 
        # Calculate the minimum and maximum values across all images 
        min_value = torch.min(images_tensor) 
        max_value = torch.max(images_tensor) 
        # Normalize the tensor using min-max normalization 
        normalized_tensor = (images_tensor - min_value) / (max_value - min_value) 
        return normalized_tensor


    def Generate(self, T_x, T_y,):
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}BeforeX.pt')
        torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}BeforeY.pt')
        self.carbonTracker.epoch_start()    
        for k in range(self.k):
            loss_avg = 0
            self.optimizerS.zero_grad()
            #Sample paratameters for network. 
            self.model._init_weights()
            loss = 0 
            # images_real_all = []
            # images_syn_all = []
            if k % 5 == 0:
                printout = True
            else: printout = False
            for c in range(self.c):
                if printout: 
                    print('Create Mini Batches')
                if c == 0:
                    T_DataX = (T_x[:Tchange_class_index])
                    S_DataX = (self.S_x[:Schange_class_index])
                    #sample w_c - omega for every class
                else:
                    T_DataX = (T_x[Tchange_class_index:])
                    S_DataX = (self.S_x[Schange_class_index:])
                    #sample w_c - omega for every class
                if printout:
                        print(MAGENTA +  f'\t\tSampling for class {c}... ' + RESET)
                T_BatchX = self.sampleRandom(T_DataX, batch_size = self.batch_size)
                S_BatchX = self.sampleRandom(S_DataX, batch_size = self.batch_size)                

                T_aug, S_aug = self.aug_strategy(T_BatchX, S_BatchX)
                T_output = self.model.embed(T_aug.type(torch.float32).to(self.device))
                S_output = self.model.embed(S_aug.type(torch.float32).to(self.device))
                loss += torch.sum((torch.mean(T_output, dim=0) - torch.mean(S_output, dim=0))**2)
                # images_real_all.append(T_aug) 
                # images_syn_all.append(S_aug)
            # images_real_all = torch.cat(images_real_all, dim=0)
            # images_syn_all = torch.cat(images_syn_all, dim=0)

            # T_output = self.model.embed(images_real_all.type(torch.float32).to(self.device))
            # S_output = self.model.embed(images_syn_all.type(torch.float32).to(self.device))

            # compute the loss
#            loss += torch.sum((torch.mean(T_output, dim=0) - torch.mean(S_output, dim=0))**2)
            loss.backward()
            self.optimizerS.step()
            loss_avg += loss.item()
            if printout:
                print(f'iteration [' + GREEN + f'{k}' + RESET + f'/{self.k}]\t avg Loss:' + GREEN+ f'{loss_avg /2}' + RESET)
            # backpropagation and weight update
            torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}IntermidiateX.pt')
            torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}IntermidiateY.pt')
        
        self.carbonTracker.epoch_end()
        self.carbonTracker.stop()
        return self.S_x, self.S_y
    

####### PARAMETERS #######
#Data parameters
#dataSet      = 'chest_xray'
dataset = 'Alzheimer_MRI'
datatype     = ''
costumLabel  = 'DM_LR1_k20000_40_'
homeDir = os.getcwd()
print(f'Running at "{homeDir}"...')
os.chdir(homeDir)
batch_size   = 32
####### PARAMETERS #######
print('preparing training data...')
#Train data
xTrain = torch.load(f'Data/Proccesed/{dataset}/trainX.pt')
yTrain = torch.load(f'Data/Proccesed/{dataset}/trainY.pt')
#xTrain = xTrain.repeat(1, 3, 1, 1)

print('\nStaring Condensation...\n')

model = M.ConvNet2()
DM = DistributionMatching(model
                        , batchSize = batch_size
                        , syntheticSampleSize = 40 #0,1% - 4, 1% - 40, 10% - 402 30% - 1206, 50% - 
                        , k = 20000
                        , c = 2
                        , lr_S = 1 # 10(ok?) 100(good)?
                        , loss_Fun = nn.BCEWithLogitsLoss()
                        , DataSet = dataset
                        , customLabel = costumLabel)

#x, y, d = GM.Generate(xTrain, yTrain)
#GM.save_output()

x, y = DM.Generate(xTrain, yTrain)
DM.save_output()

# #print(y[0])
x = x.cpu().detach().numpy()
plt.imshow(x[0][0], cmap = 'gray')
#plt.savefig('Data/Loss_chest_xray/test/DMTest.png', dpi = 400, bbox_inches = 'tight')
plt.show()