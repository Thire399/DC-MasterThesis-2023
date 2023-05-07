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

class GradientMatching():
    def __init__(self, model, batchSize:int, syntheticSampleSize:int,
                  k:int, t:int, c:int, lr_Theta:float, lr_S:float
                  , loss_Fun, DataSet:str, customLabel: str, cT_step: int) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batchSize
        self.k   = k
        self.t   = t
        self.c   = c
        self.cT_step = cT_step
        self.lr_Theta = lr_Theta
        self.savePath = f'Data/Synthetic_{DataSet}'
        os.makedirs(f'{self.savePath}/CarbonLogs', exist_ok = True)
        self.customLabel = customLabel
        self.lr_S = lr_S
#        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 1, 128, 128)), requires_grad = True) #Totally random data
        self.S_x = nn.Parameter(torch.rand((syntheticSampleSize, 1, 28, 28)), requires_grad = True) #Totally random data
        self.S_y = Gen_Y(self.S_x.shape[0]).type(torch.float32)
        self.loss_Fun = loss_Fun
        self.carbonTracker = CarbonTracker(epochs = self.k, 
                            log_dir = self.savePath + '/CarbonLogs',
                            log_file_prefix = costumLabel + self.model._get_name(),
                            monitor_epochs = -1,
                            update_interval = 1
                            )
        print(GREEN + BOLD + f'Setup:\n\tUsing Compute: {self.device}\n\tk = {k}\n\tt = {t}\n\tc = {c}\n\tLearning Rate S: = {lr_S}',
                            f'\tLearning Rate Theta = {lr_Theta}' + RESET)

    def sigmoid(self, x):
        """Sigmoid Activation Function
            Arguments:
            x.torch.tensor
            Returns
            Sigmoid(x.torch.tensor)
        """
        return 1 / (1+torch.exp(-x))
    def Distance(self, A, B):
        _sum_ = 0
        for i in range(len(A)):
            upper = torch.dot(A[i], B[i])
            lower = torch.norm(A[i]) * torch.norm(B[i])
            frac = upper / (lower + 1e-10) #division by 0
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
                   # cleanup step. (Loss here not used)
        return grad_list
    def distance_wb(self, gwr, gws):# Taken from the paper.
        shape = gwr.shape
        if len(shape) == 4: # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2: # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return torch.tensor(0, dtype=torch.float, device=gwr.device)

        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis

    def match_loss(self, gw_syn, gw_real, args): # Taken from the paper.
        #dis = torch.tensor(0.0).to(self.device)

        if args == 'ours':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += self.distance_wb(gwr, gws)

        elif args == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

        elif args == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('unknown distance function: %s'%args.dis_metric)

        return dis
    def save_output(self, x = None, y = None, after = False) -> None:
        print(RED + 'Saving synthetic dataset...' + RESET)
        if after:
            if x == None:
                torch.save(self.S_x, f = f'{self.savePath}/{self.customLabel}AfterX.pt')
                torch.save(self.S_y, f = f'{self.savePath}/{self.customLabel}AfterY.pt')
            else:
                torch.save(x, f = f'{self.savePath}/{self.customLabel}AfterX.pt')
                torch.save(y, f = f'{self.savePath}/{self.customLabel}AfterY.pt')
            print(GREEN + f'Saved "{self.customLabel}" to "{self.savePath}"'+RESET)
        else:
            if x == None:
                torch.save(self.S_x, f = f'{self.savePath}/{self.customLabel}X.pt')
                torch.save(self.S_y, f = f'{self.savePath}/{self.customLabel}Y.pt')
            else:
                torch.save(x, f = f'{self.savePath}/{self.customLabel}X.pt')
                torch.save(y, f = f'{self.savePath}/{self.customLabel}Y.pt')
            print(GREEN + f'Saved "{self.customLabel}" to "{self.savePath}"'+RESET)
        return None

    def Generate(self, T_x, T_y):
        #Note: assumes binary classification.
        Schange_class_index = torch.argmax(self.S_y).item()
        Tchange_class_index = torch.argmax(T_y).item()
        
        torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}BeforeX.pt')
        torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}BeforeY.pt')
        DistanceLst = []
        T_y = T_y.type(torch.float32)
        for k in range(self.k):
            self.carbonTracker.epoch_start()
            print(YELLOW + 'init random weights...' + RESET)
            self.model._init_weights()
            self.optimizerT = optim.SGD(self.model.parameters(), lr = self.lr_Theta, momentum = 0.5)
            self.optimizerS = optim.SGD([self.S_x], lr = self.lr_S, momentum = 0.5)
            for t in range(self.t):
                if t % 5 == 0:
                    printout = True
                    print(YELLOW + BOLD + f'K Iteration: {k}\n\tT Iteration: {t}' + RESET)
                else: printout = False
                s_loss = torch.tensor(0.0).to(self.device)
                for c in range(self.c):
                    if printout:
                        print(MAGENTA + '\t\tGenerating Batches...' + RESET )
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
                        print(f'\t\t\tSampling for class:' + RED + f' {c}...' + RESET)
                    T_BatchX = self.sampleRandom(T_DataX, batch_size = self.batch_size)
                    T_BatchY = self.sampleRandom(T_DataY, batch_size = self.batch_size)
                    S_BatchX = self.sampleRandom(S_DataX, batch_size = self.batch_size)                
                    S_BatchY = self.sampleRandom(S_DataY, batch_size = self.batch_size)
                    # Testing out paper def.
                    output_real = self.model(T_BatchX.to(self.device))
                    output_real = output_real.flatten()
                    loss_real = self.loss_Fun(output_real, T_BatchY.to(self.device))
                    
                    gw_real = torch.autograd.grad(loss_real, list(self.model.parameters()))
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    # del loss_real
                    # del T_BatchY
                    # del T_BatchX
                    # del output_real
                    # gc.collect()
                    output_syn = self.model(S_BatchX.to(self.device))
                    output_syn = output_syn.flatten()
                    loss_syn = self.loss_Fun(output_syn, S_BatchY.to(self.device))
                    gw_syn = torch.autograd.grad(loss_syn, list(self.model.parameters()), create_graph=True)
                    s_loss = self.match_loss(gw_syn, gw_real, 'cos')
                    # del S_BatchX
                    # del S_BatchY
                    # del loss_syn
                    # del gw_syn
                    # del gw_real
                    # gc.collect()
                #self.optimizerS.zero_grad()
                s_loss.backward()
                self.optimizerS.step()
                DistanceLst.append(s_loss.item())
                #     t_grad = self.GetGradient(T_BatchX, T_BatchY)
                #     s_grad = self.GetGradient(S_BatchX, S_BatchY)
                #     self.optimizerS.zero_grad()
                    # D = self.Distance(t_grad, s_grad)
                    # D.backward()

                    #self.optimizerS.step()
                    #DistanceLst.append(D.detach().cpu().numpy())

                Whole_S = torch.utils.data.TensorDataset(self.S_x, self.S_y)
                S_loader = torch.utils.data.DataLoader(Whole_S
                                                        , batch_size = self.batch_size
                                                        , shuffle = True
                                                        , num_workers = 0)
                tempLossLst = []
                if printout:
                    print(RED +'Training on whole S...'+RESET)
                self.model.train()
                # Training on whole S
                for idk in range(self.cT_step):
                    
                    if idk%25 == 0 and printout:
                        print(GREEN + f'C-inner: '+ RESET + RED + f'{idk}/{self.cT_step}'+ RESET)
                    for batch, (data, target) in enumerate(S_loader, 1):
                        #self.optimizerT.zero_grad() # a clean up step for PyTorch
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
                # if s_loss == 0 or np.mean(tempLossLst) < 1e-7: # if distance is optimal, sorta early stopping
                #     break
            torch.save(self.S_x, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}IntermidiateX.pt')
            torch.save(self.S_y, f = f'Data/Synthetic_Alzheimer_MRI/{self.customLabel}IntermidiateY.pt')
            #with torch.no_grad():
            #    self.S_x.sigmoid_()
            self.carbonTracker.epoch_end()
        self.carbonTracker.stop()
        self.S_x = self.sigmoid(self.S_x)
        return self.S_x, self.S_y, DistanceLst



####### PARAMETERS #######
#Data parameters
#dataSet      = 'chest_xray'
dataset = 'Alzheimer_MRI'
datatype     = ''
costumLabel  = 'TEST'
homeDir = os.getcwd()
print(GREEN  +f'Running at "{homeDir}"...' + RESET)
os.chdir(homeDir)
batch_size   = 32
####### PARAMETERS #######
print(RED + 'Preparing training data...' + RESET)
#Train data

########################################## TESTING MNIST DATA #########################################################################
import torch
from torchvision import datasets, transforms

# Set the path where the MNIST data will be stored

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image data
])

# Download and load the MNIST training set
train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)

# Filter the training dataset to include only class 0 and 1
train_dataset_filtered = torch.utils.data.Subset(train_dataset, [i for i, (_, label) in enumerate(train_dataset) if label == 0 or label == 1])
train_data = torch.stack([data for data, label in train_dataset_filtered])
train_labels = torch.tensor([label for _, label in train_dataset_filtered])
sorted_indices = torch.argsort(train_labels)
xTrain = train_data[sorted_indices]
yTrain = train_labels[sorted_indices]
# Create data loaders for batch processing

###################################################################################################################
# xTrain = torch.load(f'Data/Proccesed/{dataset}/trainX.pt')
# yTrain = torch.load(f'Data/Proccesed/{dataset}/trainY.pt')
# #xTrain = xTrain.repeat(1, 3, 1, 1)

print(RED + '\nStaring Condensation...\n' + RESET )
torch.manual_seed(0)
model = M.ConvNet()
GM = GradientMatching(model
                        , batchSize = batch_size #batch for updating model.
                        , syntheticSampleSize = 20
                        , k = 1
                        , t = 10
                        , c = 2
                        , cT_step = 50
                        , lr_Theta = 0.01
                        , lr_S = 0.1
                        , loss_Fun = nn.BCEWithLogitsLoss()
                        , DataSet = dataset
                        , customLabel = costumLabel)


x, y, d = GM.Generate(xTrain, yTrain)
GM.save_output(after = True)
torch.save(d, f=f'Data/Synthetic_Alzheimer_MRI/{costumLabel}testDistance.pt')

#x = x.cpu().detach().numpy()
#plt.plot(range(len(d)), d)
#plt.show()


#print(y[0])
#plt.imshow(x[0][0], cmap = 'gray')
##plt.savefig('Data/Loss_chest_xray/test/Test.png', dpi = 400, bbox_inches = 'tight')
#plt.show()
