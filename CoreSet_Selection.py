import random as rand
import numpy as np
import Models as M
import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
import gc

def RandomSelection(fileNames = None, k = 200):
    '''
    Selects k random filenames. Default 200.
    '''
    try:
        rand.seed(1)
        return rand.choices(fileNames, k = k, )
    except:
        print('Random selection: Failed')

# loop for feature extraction
def featureExtract(train_set):
    temp_list = []

    model = M.resnet(output_layer = 'layer4')
    model.eval()
    model = model.to('cuda:0')
    print('Beginning extraction...')
    #for batch, (data, target) in enumerate(train_Loader, 1):

    for batch, data in enumerate(train_set, 1):
        out = model(data.to('cuda:0').unsqueeze(0))
        temp_list.append(out.detach().cpu().numpy().flatten())
        if batch % 100 == 0:
            print('Extracting... [{}/{}]'.format(
                batch+1, len(train_set)))
    print('Finished exracting')
    feature = torch.tensor(np.asarray(temp_list))
    return feature

def getKNearest(features, dataset, k = 200, chunk = 50):
    '''Dataset = image tensor
        Features = extracted features
        # TODO rewrite documentation
    '''
    print('Running kmeans')
    kmeans = KMeans(n_clusters = 1, random_state=0, init="k-means++").fit(features)
    center = torch.from_numpy(kmeans.cluster_centers_)
    #print('\n\n' + str(features.shape) + '    ' + str(type(features)) + '\n\n\n' )
    del kmeans
    gc.collect()
    #print(center.shape, type(center))
    #TODO:Given by Julian.
    print('Calculating similarity...')
    temp = []
    for i in range(len(features)):
        temp_features = features[i:i+chunk]
        sim = F.cosine_similarity(temp_features, center.view(1, -1), dim=1)
        j = torch.argsort(sim)
        temp.append(j[:10])
        i = i + chunk
    out = [dataset[i] for i in temp[:k]]
    out = torch.stack(out)
    return out

#TODO: implement k-means either from sklearn, for kmeans coreset selection.
