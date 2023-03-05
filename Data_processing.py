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
normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
normalRandSelection = CS.RandomSelection(normal, k = trainSize)
normalTensor = SaveToTensor(normalRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
torch.save(normalTensor, f = 'Proccesed/chest_xray/train/Randomtrainnormal.pt')

pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA')
pneumoniaRandSelection = CS.RandomSelection(pneumonia, k = trainSize)
pneumoniaTensor = SaveToTensor(pneumoniaRandSelection, 64, 64)
os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/Randomtrainpneumonia.pt')

x, y = DataPrep('Proccesed/chest_xray/train/Randomtrainnormal.pt', 'Proccesed/chest_xray/train/Randomtrainpneumonia.pt')
torch.save(x, f = 'Proccesed/chest_xray/RandomtrainX.pt')
torch.save(y, f = 'Proccesed/chest_xray/RandomtrainY.pt')

print('Made Random selection dataset')
if (os.path.isfile('Proccesed/chest_xray/trainX.pt') == False) and (os.path.isfile('Proccesed/chest_xray/trainY.pt') == False):
    print('preparing training data...')
    #Train data
    normal = GetFileNames('UnProccesed/chest_xray/train/NORMAL')
    normalTensor = SaveToTensor(normal, 64, 64)
    os.makedirs('Proccesed/chest_xray/train', exist_ok = True) 
    torch.save(normalTensor, f = 'Proccesed/chest_xray/train/trainnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/train/PNEUMONIA')
    pneumoniaTensor = SaveToTensor(pneumonia, 64, 64)
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/train/trainpneumonia.pt')

    #Concat the two types
    x, y = DataPrep('Proccesed/chest_xray/train/trainnormal.pt', 'Proccesed/chest_xray/train/trainpneumonia.pt')
    torch.save(x, f = 'Proccesed/chest_xray/trainX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/trainY.pt')
    print('preparing Validation data...')
    #ValData
    normal = GetFileNames('UnProccesed/chest_xray/val/NORMAL')
    normalTensor = SaveToTensor(normal, 64, 64)
    os.makedirs('Proccesed/chest_xray/val', exist_ok = True) 
    torch.save(normalTensor, f = 'Proccesed/chest_xray/val/valnormal.pt')

    pneumonia = GetFileNames('UnProccesed/chest_xray/val/PNEUMONIA')
    pneumoniaTensor = SaveToTensor(pneumonia, 64, 64) 
    torch.save(pneumoniaTensor, f = 'Proccesed/chest_xray/val/valpneumonia.pt')
    x, y = DataPrep('Proccesed/chest_xray/val/valnormal.pt', 'Proccesed/chest_xray/val/valpneumonia.pt')
    torch.save(x, f = 'Proccesed/chest_xray/valX.pt')
    torch.save(y, f = 'Proccesed/chest_xray/valY.pt')
else:
    print('Already made val and training data.')


print('Made Train and Val set.')


## making coreset selection based on destribution
#prep the labels for normal tensor
y = LabelPrep(0, normalTensor.size()[0])
normal_Set = torch.utils.data.TensorDataset(normalTensor, y)
normal_Loader = torch.utils.data.DataLoader(normal_Set,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 0)
normalFeatures = CS.featureExtract(normal_Loader)




#y = LabelPrep(1, pneumoniaTensor.size()[0])
#pneumonia_Set = torch.utils.data.TensorDataset(normalTensor, y)
#pneumonia_Loader = torch.utils.data.DataLoader(normal_Set,
#                                            batch_size = 1,
#                                            shuffle = False,
#                                            num_workers = 0)
#pneumoniaFeatures = CS.featureExtract(normal_Loader)



### For testing -dev.
# img = Image.open('UnProccesed/chest_xray/train/PNEUMONIA\person1482_bacteria_3870.jpeg')
# img.show()
# print(np.array(img).shape)
# print(np.array(img))
# newimg = img.convert(mode = 'L')
# newimg.show()
# print(np.array(newimg).shape)
# print(np.array(newimg))

