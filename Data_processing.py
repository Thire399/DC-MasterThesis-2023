''' Data processing file '''
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import torch
import CoreSet_Selection as CS
from PIL import Image
import gc
import re
from sklearn.model_selection import train_test_split
# Change for your own file path. Should only need to change this path. all other paths should be fine as is.!!

# TODO: update function description to standard.
directory = '/home/thire399/Documents/DC-MasterThesis-2023/Data'
os.chdir(directory)
trainSize = 400
valSize = 100
imgSize = (64, 64)
vira = True


#Dataset to create
createANew = False
generateRandom = False
generateDistriution = True


#function to get all file names in the folder.
def GetFileNames(path = 'None', isVira = False):
    try:
        if isVira:
            return [os.path.join(path, file) for file in os.listdir(path) if re.search('_virus_', file)]
        else:
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
    k = 0
    for i in images:
        if k % 50 == 0:
            print(f'iteration {k}/{len(images)}')
        img = Image.open(i)
        newImg = np.array(img.resize((h, w)).convert('L')).astype(np.float32) #resizes and converts to grayscale
        #, Image.ANTIALIAS) #'Provides smothing' should not perform this probably.
        imgList.append(torch.unsqueeze(torch.from_numpy(newImg), 0)) # ndarray -> tensor and adding 1 dimension (C x H x W)
        k += 1
    return torch.stack(imgList)

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


############ MAIN ##############


# ----- Train data (RANDOM SELECTION) -----
if generateRandom == True:
    normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL', isVira= vira)
    normalRandSelection = CS.RandomSelection(normal, k = trainSize)
    normalTensor = SaveToTensor(normalRandSelection, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/train', exist_ok = True)
    torch.save(normalTensor, f = 'Proccesed/chest_xray/train/Randomtrainnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA')
    pneumoniaRandSelection = CS.RandomSelection(pneumonia, k = trainSize)
    pneumoniaTensor = SaveToTensor(pneumoniaRandSelection, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/Randomtrainpneumonia.pt')

    x, y = DataPrep('Proccesed/chest_xray/train/Randomtrainnormal.pt', 'Proccesed/chest_xray/train/Randomtrainpneumonia.pt')
    if vira:
        trainX, tempValX, trainY, tempValY = train_test_split(x, y, test_size = 0.2, random_state= 1)
        torch.save(trainX, f = 'Proccesed/chest_xray/RandomViratrainX.pt')
        torch.save(trainY, f = 'Proccesed/chest_xray/RandomViratrainY.pt')
        torch.save(tempValX, f = 'Proccesed/chest_xray/RandomViratempValX.pt')
        torch.save(tempValY, f = 'Proccesed/chest_xray/RandomViratempValY.pt')
    else:
        trainX, tempValX, trainY, tempValY = train_test_split(x, y, test_size = 0.2, random_state= 1)
        torch.save(trainX, f = 'Proccesed/chest_xray/RandomtrainX.pt')
        torch.save(trainY, f = 'Proccesed/chest_xray/RandomtrainY.pt')
        torch.save(tempValX, f = 'Proccesed/chest_xray/RandomtempValX.pt')
        torch.save(tempValY, f = 'Proccesed/chest_xray/RandomtempValY.pt')

    #ValData
    normal = GetFileNames('UnProccesed/chest_xray/val/NORMAL')
    normalTensor = SaveToTensor(normal, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/val', exist_ok = True)
    torch.save(normalTensor, f = 'Proccesed/chest_xray/val/valnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/val/PNEUMONIA')
    pneumoniaTensor = SaveToTensor(pneumonia, imgSize[0], imgSize[1])
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/val/valpneumonia.pt')
    x, y = DataPrep('Proccesed/chest_xray/val/valnormal.pt', 'Proccesed/chest_xray/val/valpneumonia.pt')
    torch.save(x, f = 'Proccesed/chest_xray/valX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/valY.pt')

    print('Made Random selection dataset')


if ((os.path.isfile('Proccesed/chest_xray/trainX.pt') == False) and (os.path.isfile('Proccesed/chest_xray/trainY.pt') == False)) or createANew == True:
    print('preparing training data...')
    #Train data
    normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
    normalTensor = SaveToTensor(normal, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/train', exist_ok = True)
    torch.save(normalTensor, f = 'Proccesed/chest_xray/train/trainnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA', isVira= vira)
    pneumoniaTensor = SaveToTensor(pneumonia, imgSize[0], imgSize[1])
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/trainpneumonia.pt')

    #Concat the two types
    x, y = DataPrep('Proccesed/chest_xray/train/trainnormal.pt', 'Proccesed/chest_xray/train/trainpneumonia.pt')
    if vira:
        trainX, tempValX, trainY, tempValY = train_test_split(x, y, test_size = 0.2, random_state= 1)
        torch.save(trainX, f = 'Proccesed/chest_xray/ViratrainX.pt')
        torch.save(trainY, f = 'Proccesed/chest_xray/ViratrainY.pt')
        torch.save(tempValX, f = 'Proccesed/chest_xray/ViratempValX.pt')
        torch.save(tempValY, f = 'Proccesed/chest_xray/ViratempValY.pt')

    else:
        trainX, tempValX, trainY, tempValY = train_test_split(x, y, test_size = 0.2, random_state= 1)
        torch.save(trainX, f = 'Proccesed/chest_xray/trainX.pt')
        torch.save(trainY, f = 'Proccesed/chest_xray/trainY.pt')
        torch.save(tempValX, f = 'Proccesed/chest_xray/tempValX.pt')
        torch.save(tempValY, f = 'Proccesed/chest_xray/tempValY.pt')

    print('preparing Validation data...')
    #ValData
    normal = GetFileNames('UnProccesed/chest_xray/val/NORMAL')
    normalTensor = SaveToTensor(normal, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/val', exist_ok = True)
    torch.save(normalTensor, f = 'Proccesed/chest_xray/val/valnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/val/PNEUMONIA')
    pneumoniaTensor = SaveToTensor(pneumonia, imgSize[0], imgSize[1]) 
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/val/valpneumonia.pt')
    x, y = DataPrep('Proccesed/chest_xray/val/valnormal.pt', 'Proccesed/chest_xray/val/valpneumonia.pt')
    torch.save(x, f = 'Proccesed/chest_xray/valX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/valY.pt')
    print('Made Train and Val set.')
#else:
#    print('Already made full val and training data.')


if generateDistriution == True:
    print('\n\nStarting coreset selection distribution.')
    ## making coreset selection based on destribution
    #prep the labels for normal tensor
    print('Non-sick images...')
    normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
    kmeansNormalTensor = SaveToTensor(normal, h = 800, w = 800)
    #kmeansNormalTensor = kmenasNormalTensor.repeat(1,3,1,1) #recreates the tensor to (n, 3, 64, 64)
    normalFeatures = CS.featureExtract(kmeansNormalTensor)
    normalDistribution = CS.getKNearest(normalFeatures, normal, 400)
    del kmeansNormalTensor
    del normalFeatures
    gc.collect()
    normalTensor = SaveToTensor(normalDistribution, imgSize[0], imgSize[1])
    torch.save(normalDistribution, f = 'Proccesed/chest_xray/train/normalDistribution.pt')
    del normalTensor
    del normalDistribution
    gc.collect()

    ## making coreset selection based on destribution
    #prep the labels for pneumonia tensor
    print('Sick images...')
    pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA', isVira = vira)
    kmeansPneumoniaTensor = SaveToTensor(pneumonia, h = 800, w = 800)
    pneumoniaFeatures = CS.featureExtract(kmeansPneumoniaTensor)
    del kmeansPneumoniaTensor
    gc.collect()
    pneumoniaDistribution = CS.getKNearest(pneumoniaFeatures, pneumonia, 400)
    pneumoniaTensor = SaveToTensor(pneumoniaDistribution, imgSize[0], imgSize[1])
    torch.save(pneumoniaTensor , f = 'Proccesed/chest_xray/train/pneumoniaDistribution.pt')
    del pneumoniaTensor
    gc.collect()
    print('Saving tensor...')

    x, y = DataPrep('Proccesed/chest_xray/train/normalDistribution.pt', 'Proccesed/chest_xray/train/pneumoniaDistribution.pt')
    torch.save(x, f = 'Proccesed/chest_xray/DistributionTrainX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/DistributionTrainY.pt')
    print('Done.')
    #ValData
    print('\nRegenerating validation data...')
    normal = GetFileNames('UnProccesed/chest_xray/val/NORMAL')
    normalTensor = SaveToTensor(normal, imgSize[0], imgSize[1])
    os.makedirs('Proccesed/chest_xray/val', exist_ok = True)
    torch.save(normalTensor, f = 'Proccesed/chest_xray/val/valnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/val/PNEUMONIA')
    pneumoniaTensor = SaveToTensor(pneumonia, imgSize[0], imgSize[1])
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/val/valpneumonia.pt')
    x, y = DataPrep('Proccesed/chest_xray/val/valnormal.pt', 'Proccesed/chest_xray/val/valpneumonia.pt')
    torch.save(x, f = 'Proccesed/chest_xray/valX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/valY.pt')


### For testing -dev.
# img = Image.open('UnProccesed/chest_xray/train/PNEUMONIA\person1482_bacteria_3870.jpeg')
# img.show()
# print(np.array(img).shape)
# print(np.array(img))
# newimg = img.convert(mode = 'L')
# newimg.show()
# print(np.array(newimg).shape)
# print(np.array(newimg))

