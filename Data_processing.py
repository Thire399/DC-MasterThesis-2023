''' Data processing file '''
import numpy as np
import os 
import shutil
import torch
import CoreSet_Selection as CS
import gc
import re
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import random
# Change for your own file path. Should only need to change this path. all other paths should be fine as is.!!

# TODO: update function description to standard.
# TODO: update to be a function call in some other document, that takes all these parameters
directory = '/home/thire399/Documents/School/DC-MasterThesis-2023/Data'
os.chdir(directory)
healthy_size = 1
unhealthy_size = 3
SampleRatio = 0.3 # procentage of the dataset.
imgSize = (800, 800)
vira = False
alzimers = False
Chest_Xray = False
customLabel = '01Percent'
#Dataset to create
make_new_split = False
generateRandom = False
generateDistriution = False
Val = False # only for chest x_ray
os.makedirs('Proccesed/chest_xray/train', exist_ok = True)
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

def SaveToTensor(images, h = 32, w = 32, reshape = True):
    ''' should take in a image and resize it tot h x w (keep layers?). TODO: update description'''
    imgList = []
    k = 0
    for i in images:
        if k % 50 == 0:
            print(f'iteration {k}/{len(images)}')
        img = Image.open(i)
        if reshape:
           newImg = np.array(img.resize((h, w)).convert('L')).astype(np.float32)/255#resizes and converts to grayscale
        else:
            newImg = np.array(img.convert('L')).astype(np.float32)/255
        #, Image.ANTIALIAS) #'Provides smothing' should not perform this probably.
        imgList.append(torch.unsqueeze(torch.from_numpy(newImg), 0)) # ndarray -> tensor and adding 1 dimension (C x H x W)
        k += 1
    return torch.stack(imgList)

def ReadFromTensor(pathToTensor, h, w, readfromFile = True):
    if readfromFile:
        input = torch.load(pathToTensor)
        print(f'Loading from saved tensor: {pathToTensor}...')
    else:
        input = pathToTensor
    imgList = []
    for i in range(input.shape[0]):
        if i % 50 == 0:
            print(f'(resize) iteration {i}/{input.shape[0]}')
            pil_image = T.ToPILImage()(input[i])
            newImg = np.array(pil_image.resize((h, w)))
        imgList.append(torch.unsqueeze(torch.from_numpy(newImg), 0))
    return torch.stack(imgList)

def split(listOfPaths, TrainingSampleSize: float = 0.7):
    k = int(np.rint(len(listOfPaths)*TrainingSampleSize))
    print(f'listofPaths size: {len(listOfPaths)}\n splitsize {TrainingSampleSize}\nnp.rint size: {np.rint(len(listOfPaths)*TrainingSampleSize)}\nk: {k} ')
    print('plit size (choosen)', k)
    choosen = random.sample(listOfPaths, k = k, )
    rest = [elem for elem in listOfPaths if elem not in choosen]
    print('residual size: ', len(rest))
    return choosen, rest

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
if Chest_Xray:
    if make_new_split:
        print('Cerating a new training split...')
        normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
        newNormal, valNormalTensor = split(normal)
        normalTensor = SaveToTensor(newNormal, 800, 800)
        torch.save(normalTensor, f = 'Proccesed/chest_xray/train/Splitnormal.pt')
        del normal
        del newNormal
        del normalTensor
        gc.collect()
        pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA')
        newPneumonia, valPneumoniaTensor = split(pneumonia)
        pneumoniaTensor = SaveToTensor(newPneumonia, 800, 800)
        torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/Splitpneumonia.pt')
        del pneumonia
        del newPneumonia
        del pneumoniaTensor
        gc.collect
        x, y = DataPrep('Proccesed/chest_xray/train/Splitnormal.pt', 'Proccesed/chest_xray/train/Splitpneumonia.pt')
        #x = ReadFromTensor(x, imgSize[0], imgSize[1], readfromFile = False)
        torch.save(x, f = 'Proccesed/chest_xray/trainX.pt')
        torch.save(y, f = 'Proccesed/chest_xray/trainY.pt')
        del x
        del y
        gc.collect()
        print('Validation data...')
        
        valNormalTensor = SaveToTensor(valNormalTensor, imgSize[0], imgSize[1])
        valPneumoniaTensor = SaveToTensor(valPneumoniaTensor, imgSize[0], imgSize[1])

        torch.save(valNormalTensor, f = 'Proccesed/chest_xray/train/tempNormalValX.pt')
        torch.save(valPneumoniaTensor, f = 'Proccesed/chest_xray/train/tempPneumoniaValX.pt')
        x, y = DataPrep('Proccesed/chest_xray/train/tempNormalValX.pt', 'Proccesed/chest_xray/train/tempPneumoniaValX.pt')
        torch.save(x, f = 'Proccesed/chest_xray/tempValX.pt')
        torch.save(y, f = 'Proccesed/chest_xray/tempValY.pt')
        print('Created a new data split.')
        if (os.path.isfile('Proccesed/chest_xray/valX.pt') == False) and (os.path.isfile('Proccesed/chest_xray/valY.pt') == False) or Val:
            print('Generating Original Validation set...')
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
            print('Generated Original Validation set')

    if generateRandom == True:
        print('Generating random...')
        X = torch.load('Proccesed/chest_xray/train/Splitnormal.pt')
        #X = ReadFromTensor(X, imgSize[0], imgSize[1], readfromFile= False)
        #Y = torch.load('Proccesed/chest_xray/trainY.pt')
        Y = torch.from_numpy(np.asarray([0]*X.shape[0]))
        NormalRandX, NormalRandY = CS.RandomSelection(X, Y, k = healthy_size)
        #Unhealthy
        X = torch.load('Proccesed/chest_xray/train/Splitpneumonia.pt')
        #X = ReadFromTensor(X, imgSize[0], imgSize[1], readfromFile= False)
        #Y = torch.load('Proccesed/chest_xray/trainY.pt')
        Y = torch.from_numpy(np.asarray([1]*X.shape[0]))
        pneumoniaRandX, pneumoniaRandY = CS.RandomSelection(X, Y, k = unhealthy_size)
        RandX = torch.cat((NormalRandX, pneumoniaRandX))
        RandY = torch.cat((NormalRandY, pneumoniaRandY))
        torch.save(RandX, f'Proccesed/chest_xray/{customLabel}RandomtrainX.pt')
        torch.save(RandY, f'Proccesed/chest_xray/{customLabel}RandomtrainY.pt')
        del pneumoniaRandX
        del pneumoniaRandY
        del NormalRandX
        del NormalRandY
        del X
        del Y
        gc.collect()
        if (os.path.isfile('Proccesed/chest_xray/valX.pt') == False) and (os.path.isfile('Proccesed/chest_xray/valY.pt') == False):
            print('Generating Original Validation set...')
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
            print('Generated Original Validation set')
        print('Made Random selection dataset')      

    if generateDistriution == True:
        print('\n\nStarting coreset selection distribution.')
        ## making coreset selection based on destribution
        #prep the labels for normal tensor
        print('Non-sick images...')
        normal = torch.load('Proccesed/chest_xray/train/Splitnormal.pt')
        normalFeatures = CS.featureExtract(normal)
        #normal = ReadFromTensor('Proccesed/chest_xray/train/Splitnormal.pt',
        #                             imgSize[0], imgSize[1])
        normalDistribution = CS.getKNearest(normalFeatures, normal, healthy_size)
        del normal
        del normalFeatures
        gc.collect()
        torch.save(normalDistribution, f = 'Proccesed/chest_xray/train/normalDistribution.pt')
        del normalDistribution
        gc.collect()

        ## making coreset selection based on destribution
        #prep the labels for pneumonia tensor
        print('Sick images...')
        pneumonia = torch.load('Proccesed/chest_xray/train/Splitpneumonia.pt')
        pneumoniaFeatures = CS.featureExtract(pneumonia)
        gc.collect()
        #pneumonia = ReadFromTensor('Proccesed/chest_xray/train/Splitpneumonia.pt',
        #                        imgSize[0], imgSize[1])
        pneumoniaDistribution = CS.getKNearest(pneumoniaFeatures, pneumonia, unhealthy_size)
        torch.save(pneumoniaDistribution , f = 'Proccesed/chest_xray/train/pneumoniaDistribution.pt')
        del pneumoniaFeatures
        del pneumoniaDistribution
        gc.collect()
        print('Saving tensor...')
        x, y = DataPrep('Proccesed/chest_xray/train/normalDistribution.pt', 'Proccesed/chest_xray/train/pneumoniaDistribution.pt')
        torch.save(x, f = f'Proccesed/chest_xray/{customLabel}DistributiontrainX.pt')
        torch.save(y, f = f'Proccesed/chest_xray/{customLabel}DistributiontrainY.pt')
        print('Done.')
        del x
        del y
        gc.collect()
        #ValData
        if (os.path.isfile('Proccesed/chest_xray/valX.pt') == False) and (os.path.isfile('Proccesed/chest_xray/valY.pt') == False):
            print('Generating Original Validation set...')
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

if alzimers == True:
    
    if make_new_split:
        print('Creating Full data set...')
        os.makedirs('Proccesed/Alzheimer_MRI/train', exist_ok = True)
        print('Getting healty images...')
        normal = GetFileNames('UnProccesed/Alzheimer_MRI/Non_Demented')
        newNormal, testNormal = split(normal, TrainingSampleSize = 0.9)
        newNormal, valNormal = split(newNormal)
        normalTensor = SaveToTensor(newNormal, reshape = False)
        valnormalTensor = SaveToTensor(valNormal, reshape = False)
        testnormalTensor = SaveToTensor(testNormal, reshape = False)
        torch.save(normalTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitTrainnormal.pt')
        torch.save(valnormalTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitValnormal.pt')
        torch.save(testnormalTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitTestnormal.pt')
        del normal # clean up step
        del newNormal
        del normalTensor
        del valnormalTensor
        del testnormalTensor
        gc.collect()
        print('Getting Sick images...')
        Moderate = GetFileNames('UnProccesed/Alzheimer_MRI/Moderate_Demented')
        Mild = GetFileNames('UnProccesed/Alzheimer_MRI/Mild_Demented')
        VeryMild = GetFileNames('UnProccesed/Alzheimer_MRI/Very_Mild_Demented')
        Demented = Moderate + Mild + VeryMild
        newDemented, testDemented = split(Demented, TrainingSampleSize = 0.9)
        newDemented, valDemented = split(newDemented)
        DementedTensor = SaveToTensor(newDemented, reshape = False)
        valDementedTensor = SaveToTensor(valDemented, reshape = False)
        testDementedTensor = SaveToTensor(testDemented, reshape = False)
        torch.save(DementedTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitTrainDemented.pt')
        torch.save(valDementedTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitValDemented.pt')
        torch.save(testDementedTensor, f = 'Proccesed/Alzheimer_MRI/train/SplitTestDemented.pt')
        del Demented # clean up step
        del newDemented
        del DementedTensor
        del valDementedTensor
        del testDementedTensor
        gc.collect()
        print('Saving Tensors to: "Proccesed/Alzheimer_MRI/"')
        x, y = DataPrep('Proccesed/Alzheimer_MRI/train/SplitTrainnormal.pt', 'Proccesed/Alzheimer_MRI/train/SplitTrainDemented.pt')
        torch.save(x, f = 'Proccesed/Alzheimer_MRI/trainX.pt')
        torch.save(y, f = 'Proccesed/Alzheimer_MRI/trainY.pt')

        x, y = DataPrep('Proccesed/Alzheimer_MRI/train/SplitValnormal.pt', 'Proccesed/Alzheimer_MRI/train/SplitValDemented.pt')
        torch.save(x, f = 'Proccesed/Alzheimer_MRI/ValX.pt')
        torch.save(y, f = 'Proccesed/Alzheimer_MRI/ValY.pt')

        x, y = DataPrep('Proccesed/Alzheimer_MRI/train/SplitTestnormal.pt', 'Proccesed/Alzheimer_MRI/train/SplitTestDemented.pt')
        torch.save(x, f = 'Proccesed/Alzheimer_MRI/testX.pt')
        torch.save(y, f = 'Proccesed/Alzheimer_MRI/testY.pt')
        print('Made full dataset')

    if generateRandom == True:
        print('Creating Random data set...')
        os.makedirs('Proccesed/Alzheimer_MRI/train', exist_ok = True)
        print('Getting healty images...')
        NormalX = torch.load('Proccesed/Alzheimer_MRI/train/SplitTrainnormal.pt')
        NormalY = torch.from_numpy(np.asarray([0]*NormalX.shape[0]))
        NormalRandX, NormalRandY = CS.RandomSelection(NormalX, NormalY, k = int(np.rint(NormalX.shape[0] * SampleRatio)))
        #torch.save(NormalRandX, f = 'Proccesed/Alzheimer_MRI/train/RandomTrainNormal.pt')
        DementedX = torch.load('Proccesed/Alzheimer_MRI/train/SplitTrainDemented.pt')
        DementedY = torch.from_numpy(np.asarray([1]*DementedX.shape[0]))
        DementedRandX, DementedRandY = CS.RandomSelection(DementedX, DementedY, k = int(np.rint(DementedX.shape[0] * SampleRatio)))        
        RandX = torch.cat((NormalRandX, DementedRandX))
        RandY = torch.cat((NormalRandY, DementedRandY))
        torch.save(RandX, f = f'Proccesed/Alzheimer_MRI/{customLabel}RandomtrainX.pt')
        torch.save(RandY, f = f'Proccesed/Alzheimer_MRI/{customLabel}RandomtrainY.pt')
        del NormalX
        del NormalY
        del NormalRandX
        del NormalRandY
        del DementedX
        del DementedY
        del DementedRandX
        del DementedRandY
        del RandX
        del RandY
        gc.collect()
        print('Made Random selection dataset')      

    if generateDistriution:
        print('\n\nStarting coreset selection distribution.')
        ## making coreset selection based on destribution
        #prep the labels for normal tensor
        print('Creating Random data set...')
        os.makedirs('Proccesed/Alzheimer_MRI/train', exist_ok = True)
        print('Getting healty images...')
        NormalX = torch.load('Proccesed/Alzheimer_MRI/train/SplitTrainnormal.pt')
        normalFeatures = CS.featureExtract(NormalX)
        normalDistribution = CS.getKNearest(normalFeatures, NormalX, k = int(np.rint(NormalX.shape[0] * SampleRatio)))
        torch.save(normalDistribution, f = 'Proccesed/Alzheimer_MRI/train/normalDistribution.pt')
        del NormalX
        del normalFeatures
        del normalDistribution
        gc.collect()
        print('Getting Sick images...')
        DementedX = torch.load('Proccesed/Alzheimer_MRI/train/SplitTrainDemented.pt')
        DementedFeatures = CS.featureExtract(DementedX)
        DementedDistribution = CS.getKNearest(DementedFeatures, DementedX, int(np.rint(DementedX.shape[0] * SampleRatio)))
        torch.save(DementedDistribution, f = 'Proccesed/Alzheimer_MRI/train/DementedDistribution.pt')
        del DementedX
        del DementedFeatures
        del DementedDistribution
        gc.collect()
        print('Generating final tensor...')
        X, Y = DataPrep('Proccesed/Alzheimer_MRI/train/normalDistribution.pt', 'Proccesed/Alzheimer_MRI/train/DementedDistribution.pt')
        torch.save(X, f = f'Proccesed/Alzheimer_MRI/{customLabel}DistributiontrainX.pt')
        torch.save(Y, f = f'Proccesed/Alzheimer_MRI/{customLabel}DistributiontrainY.pt')
        print('Done. -> Proccesed/Alzheimer_MRI/')
    if generateRandom == False and generateDistriution == False:
        print('No Version Choosen -> Nothing made')