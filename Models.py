''' This file contains the models used for baseline prediction.'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, enc_chs=(3, 8, 16, 32, 64, 128), dec_chs=(128, 64, 32, 16, 8), num_class=1, retain_dim=True): #Change num_class to handle 4 channels
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1) 
        self.retain_dim  = retain_dim
        self.sig         = nn.Sigmoid() #clamps the output to between 1 and 0
                #clamps output between 1 and 0. differently from the sigmoid
        self.num_class   = num_class #think of it as the number of objects to segment

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if (self.num_class == 1):
            out      = self.sig(out)
        else:
            out = self.softMax(out)
        if self.retain_dim: 
            out = F.interpolate(out, (x.shape[2],x.shape[3]), mode = 'nearest') #nearest #Bicubic should have less artifacts
        return out


############ ResNet? ###################

############ UNet third? ###################