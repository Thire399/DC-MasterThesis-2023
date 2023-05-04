''' This file contains the models used for baseline prediction.'''
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import warnings
warnings.filterwarnings("ignore")

############ UNet ###################
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding= 1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding= 1)
        self.BN    = nn.BatchNorm2d(out_ch, affine=False)
        self.Drop  = nn.Dropout(0.20)

    def forward(self, x):
        return self.relu(self.conv2( self.relu( self.conv1(x))))

#This is the downsampling step/ the encoding step.
class Encoder(nn.Module):
    def __init__(self, chs=(4, 8, 16, 32, 64, 128)): # was 64,128,256,512,1024
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

#This is the decoder, or where we upsample again, / putting everything together
class Decoder(nn.Module):
    def __init__(self, chs=(128, 64, 32, 16, 8)): # was 1024, 512, 256, 128, 64
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2, padding=0) for i in range(len(chs)-1)]) #maybe use torch unpool "max unpool 2D" 
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

#The final UNet implementation
#was def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 8, 16, 32, 64, 128), dec_chs=(128, 64, 32, 16, 8), num_class=1, df = 4095): #Change num_class to handle 4 channels
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.sig         = nn.Sigmoid() #clamps the output to between 1 and 0
                #clamps output between 1 and 0. differently from the sigmoid
        self.fc1 = nn.Linear(df, 64)
        self.fc2 = nn.Linear(64, num_class)
        self.num_class   = num_class #think of it as the number of objects to segment

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out      = self.sig(out)
        temp = torch.flatten(out,start_dim = 1)
        out = F.relu(self.fc1(temp))
        out = self.fc2(out)

        return out


############ ResNet? ###################

class resnet(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        self.conv = nn.Conv2d(1, 3, kernel_size= 1, stride = 1) # Kernel size 1 and stride 1 to change dimensionality.
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.conv(x)
        x = self.net(x)
        return x


########## Condensation model from the paper ##############

class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(32768, num_classes)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)#, mean=0.0, std=0.01)
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(True)

        return None
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

##########################################################


########## Condensation model from the paper 2 ##############

class ConvNet2(nn.Module):
    def __init__(self, num_classes=1, embedding_size = 32):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        #self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        #self.embedding = nn.Linear(128*3*3, num_classes)
        #self.fc = nn.Linear(embedding_size, num_classes)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)#, mean=0.0, std=0.01)
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(True)
        return None
    
    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.pool2(out)
     
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        #out = self.avgpool(out)
        #out = out.view(out.size(0), -1,1,1)
        #out = self.embedding(out)
        #out = F.normalize(out, p=2, dim=1) # L2 normalization
        #print(out[0].shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


##########################################################

######## Early Stopping ############
#The early stopping class is from here this github, and the credit goes to him.
# I only use it for early stopping.
#https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, saveModel = True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.saveModel = saveModel
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.saveModel:
            torch.save(model.state_dict(), self.path) #Custome save made.
        self.val_loss_min = val_loss