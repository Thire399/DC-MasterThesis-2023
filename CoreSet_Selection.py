import random as rand
import numpy as np
import torch
from sklearn.cluster import KMeans

def RandomSelection(fileNames = None, k = 200):
    '''
    Selects k random filenames. Default 200.
    '''
    try:
        return rand.choices(fileNames, k = k)
    except:
        print('Random selection: Failed')



# loop for feature extraction
def featureExtract(train_Loader):
    temp_list = []
    model.eval()
    model = model.to('cuda:0')
    for batch, (data, target) in enumerate(train_Loader, 1):
        out = model(data.to('cuda:0'))
        temp_list.append(out.detach().cpu().numpy().flatten())
        if batch % 100 == 0:
            print('Train images [{}/{}]'.format(
                batch+1, len(train_Loader)))
            break
    feature = torch.tensor(np.asarray(temp_list))

    return feature

# TODO rewrite to actually work
def CalDistance(x, Data):
    '''
    Parameter: x is a constant value. \n
    Parameter: Data the dataset matrix. \n
    ---------------- \n
    utalizing that ||x-y||^2 = ||x||^2+||y||^2 - 2 x^Ty
    ----------------\n
    Return: Returns the distance from x to each point from the dataset.
    '''
    distancesSqr = np.sum(x**2) + np.sum(Data**2, axis = 0) - 2 * np.transpose(x) @ Data
     #Note that my dataset is (2 x 100), therefore, we use axis = 0 instead of 1.
    distances = np.sqrt(distancesSqr) #elementwise square root.
    return distances

def Knn (D, k, x):
    '''
    Parameters: Dataset (D) \n
    Parameters: Number of nearst neighbors (k) \n
    Parameters: and a point (x) \n
    Parameters: Labels given by the target function. (the asssignment description is shiet)
    Returns: the label of point x according to the knn algorithm \n
    '''
    Dist = CalDistance(x, D)
    #sorts the distances and returns the index of the distances.
    SortDistIndex = np.argsort(Dist)
    KIndex = SortDistIndex[:k]
    out = 
    return KIndex

def getKNearest(features, dataset, k = 200):
    '''Dataset = image tensor
        Features = extracted features
        # TODO rewrite documentation
    '''
    kmeans = kmeans = KMeans(n_clusters = 1, random_state=0, init="k-means++").fit(features)
    center = kmeans.cluster_centers_
    temp = Knn(center, k = k, x = dataset)
    

#    for i in range()
    return None

#TODO: implement k-means either from sklearn, for kmeans coreset selection.
