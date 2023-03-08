import random as rand
import numpy as np
import Models as M
import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F

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
    
    model = M.resnet(output_layer = 'layer4')
    model.eval()
    model = model.to('cuda:0')
    for batch, (data, target) in enumerate(train_Loader, 1):
        out = model(data.to('cuda:0'))
        temp_list.append(out.detach().cpu().numpy().flatten())
        if batch % 100 == 0:
            print('Train images [{}/{}]'.format(
                batch+1, len(train_Loader)))
    feature = torch.tensor(np.asarray(temp_list))

    return feature

def getKNearest(features, dataset, k = 200):
    '''Dataset = image tensor
        Features = extracted features
        # TODO rewrite documentation
    '''

    kmeans = KMeans(n_clusters = 1, random_state=0, init="k-means++").fit(features)
    center = torch.from_numpy(kmeans.cluster_centers_)
    #print(center.shape, type(center))
    #TODO:Given by Julian.
    sim = F.cosine_similarity(features, center.view(1, -1), dim=1)
    j = torch.argsort(sim)
    out = [dataset[i] for i in j[:k]]
    out = torch.stack(out)
    return out

#TODO: implement k-means either from sklearn, for kmeans coreset selection.
