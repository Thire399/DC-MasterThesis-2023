import random as rand

def RandomSelection(fileNames = None, k = 200):
    '''
    Selects k random filenames. Default 200.
    '''
    try:
        return rand.choices(fileNames, k = k)
    except:
        print('Random selection: Failed')
