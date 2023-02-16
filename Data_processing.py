''' Data processing file '''
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os 
import shutil

# Change for your own file path.
#os.chdir('/Users/thire/Documents/School/DC-MasterThesis-2023/Data')

#print(os.listdir('UnProccesed/chest_xray'))

#function to get all file names in the folder.
def GetFileNames(path = 'None'):
    try: 
        fileNames = os.listdir(path)
        return fileNames
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




