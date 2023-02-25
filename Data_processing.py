''' Data processing file '''
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import torch
import CoreSet_Selection as CS
from PIL import Image
# Change for your own file path. Should only need to change this path. all other paths should be fine as is.!!

# TODO: update function description to standard.
directory = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data'
os.chdir(directory)
trainSize = 400
valSize = 100

#function to get all file names in the folder.
def GetFileNames(path = 'None'):
    try: 
        return [os.path.join(path, file) for file in os.listdir(path)]
    except:
        print('Path "{0}" Not found'.format(path))

def MoveFiles(fileNames, source, destination):
    # Note copy has odd behaviour. - Depricated -> Now using "SaveToTensor"
    #Removes all content in folder first
    for f in os.listdir(destination):
        os.remove(os.path.join(destination, f))
    # copys the files in the original folder to the new folder
    for f in fileNames:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.copy(src_path, dst_path)
    print('moved all files to "{0}"'.format(destination))
    return None

def SaveToTensor(images, h = 32, w = 32):
    ''' should take in a image and resize it tot h x w (keep layers?). TODO: update description''' 
    imgList = []
    for i in images:    
        img = Image.open(i)
        newImg = np.array(img.resize((h, w)).convert('L')) #resizes and converts to grayscale
        if newImg.shape != (64, 64):
            print(i) #, Image.ANTIALIAS) #'Provides smothing' should not perform this probably.
        imgList.append(torch.unsqueeze(torch.from_numpy(newImg), 0)) # ndarray -> tensor and adding 1 dimension (C x H x W) 
    return torch.stack(imgList)
    

############ MAIN ##############

# ----- Train data -----
normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
normalRandSelection = CS.RandomSelection(normal, k = trainSize)
normalTensor = SaveToTensor(normalRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
torch.save(normalTensor, f = 'Proccesed/chest_xray/train/trainnormal.pt')
print('Saved "normal.pt" to \n"{0}Proccesed/chest_xray/train"\n{1} images saved.'.format(directory, normalTensor.shape[0]))

pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA')
pneumoniaRandSelection = CS.RandomSelection(pneumonia, k = trainSize)
pneumoniaTensor = SaveToTensor(pneumoniaRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/trainpneumonia.pt')
print('Saved "pneumonia.pt" to \n"{0}Proccesed/chest_xray/train"\n{1} images saved.'.format(directory, pneumoniaTensor.shape[0]))

# ---- Validation data ------
normal = GetFileNames('UnProccesed/chest_xray/val/NORMAL')
normalRandSelection = CS.RandomSelection(normal, k = valSize)
normalTensor = SaveToTensor(normalRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/val', exist_ok = True) 
torch.save(normalTensor, f = 'Proccesed/chest_xray/val/valnormal.pt')
print('Saved "valnormal.pt" to \n"{0}Proccesed/chest_xray/val"\n{1} images saved.'.format(directory, normalTensor.shape[0]))

pneumonia = GetFileNames('UnProccesed/chest_xray/val/PNEUMONIA')
pneumoniaRandSelection = CS.RandomSelection(pneumonia, valSize)
pneumoniaTensor = SaveToTensor(pneumoniaRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/val', exist_ok = True) 
torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/val/valpneumonia.pt')
print('Saved "valpneumonia.pt" to \n"{0}Proccesed/chest_xray/val"\n{1} images saved.'.format(directory, pneumoniaTensor.shape[0]))


############## combining the tensors.

def LabelPrep(label, k = 200):
    tempL = np.asarray([label]*k)
    return torch.tensor(tempL)
    

def DataPrep (class1, class2):
    ''' Should take in both classes of train and concat them'''
    normalX = torch.load(f = class1)
    pneumoniaX = torch.load(f = class2)
    nY = LabelPrep(label = 0, k = normalX.size()[0])
    pY = LabelPrep(label = 1, k = pneumoniaX.size()[0])

    x = torch.cat((normalX, pneumoniaX))
    y = torch.cat((nY, pY))
    return x, y

x, y = DataPrep('Proccesed/chest_xray/train/trainnormal.pt', 'Proccesed/chest_xray/train/trainpneumonia.pt')

torch.save(x, f = 'Proccesed/chest_xray/trainX.pt')
torch.save(x, f = 'Proccesed/chest_xray/trainY.pt')

x, y = DataPrep('Proccesed/chest_xray/val/valnormal.pt', 'Proccesed/chest_xray/val/valpneumonia.pt')
torch.save(x, f = 'Proccesed/chest_xray/valX.pt')
torch.save(x, f = 'Proccesed/chest_xray/valY.pt')

print('Made Train and Val set.')

### For testing -dev. 
# img = Image.open('UnProccesed/chest_xray/train/PNEUMONIA\person1482_bacteria_3870.jpeg')
# img.show()
# print(np.array(img).shape)
# print(np.array(img))
# newimg = img.convert(mode = 'L')
# newimg.show()
# print(np.array(newimg).shape)
# print(np.array(newimg))



