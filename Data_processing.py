''' Data processing file '''
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os 
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
# Change for your own file path.
directory = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data'
os.chdir(directory)

#print(os.listdir('UnProccesed/chest_xray'))

#function to get all file names in the folder.
def GetFileNames(path = 'None'):
    try: 
        return [os.path.join(path, file) for file in os.listdir(path)]
    except:
        print('Path "{0}" Not found'.format(path))

def RandomSelection(fileNames = None, k = 200):
    '''
    Selects k random filenames. Default 200.
    '''
    try:
        return rand.choices(fileNames, k = k)
    except:
        print('Random selection: Failed')


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

def SaveToTensor(images,h = 32, w = 32):
    ''' should take in a image and resize it tot h x w (keep layers?). TODO: update description''' 
    tempImg = []
    k = 0
    for i in images:    
        img = Image.open(i)
        k += 1
        newImg = np.array(img.resize((h, w))) #, Image.ANTIALIAS) #'Provides smothing' should not perform this probably.
        tempImg.append(torch.unsqueeze(torch.from_numpy(newImg), 0)) # ndarray -> tensor and adding 1 dimension (C x H x W) 
    return torch.stack(tempImg)
    
############ MAIN ##############

normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
normalRandSelection = RandomSelection(normal)
normalTensor = SaveToTensor(normalRandSelection)
torch.save(normalTensor, f = 'Proccesed/chest_xray/train/normal.pt')
print('Saved "normal.pt" to \n"{0}Proccesed/chest_xray/train"\n{1} images saved.'.format(directory, normalTensor.shape[0]))


#normalSource = 'UnProccesed/chest_xray/train/NORMAL'
#normalDestination = 'Proccesed/chest_xray/train/NORMAL'
#MoveFiles(normalRandSelection, normalSource, normalDestination)


'''
normal = GetFileNames('/Users/thire/Documents/School/DC-MasterThesis-2023/Data/UnProccesed/chest_xray/train/NORMAL')
normalRandSelection = RandomSelection(normal)
print(len(normalRandSelection))
normalSource = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data/UnProccesed/chest_xray/train/NORMAL'
normalDestination = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data/Proccesed/chest_xray/train/NORMAL'
MoveFiles(normalRandSelection, normalSource, normalDestination)

pneumonia = GetFileNames('/Users/thire/Documents/School/DC-MasterThesis-2023/Data/UnProccesed/chest_xray/train/PNEUMONIA')
pneumoniaRandSelection = RandomSelection(pneumonia)
print(len(pneumoniaRandSelection))
pneumoniaSource = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data/UnProccesed/chest_xray/train/PNEUMONIA'
pneumoniaDestination = '/Users/thire/Documents/School/DC-MasterThesis-2023/Data/Proccesed/chest_xray/train/PNEUMONIA'
MoveFiles(pneumoniaRandSelection, pneumoniaSource, pneumoniaDestination)
'''



