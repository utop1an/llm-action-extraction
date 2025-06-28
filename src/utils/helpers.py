import pickle
import numpy as np
import os

def load_pkl(path):
    """
    Load pickle file
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj
    
def ten_fold_split_ind(num_data, fname, k, random=True):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        tmp_inds = np.arange(num_data)
        np.random.shuffle(tmp_inds)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_inds[i*n: ])
            else:
                indices.append(tmp_inds[i*n: (i+1)*n])
    else:
        for i in range(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices



def index2data(indices, data):
    """
    Obtain k-fold data according to given indices
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
        num_data = len(data[keys[0]])
        for i in range(len(indices)):
            valid_data = {}
            train_data = {}
            for k in keys:
                valid_data[k] = []
                train_data[k] = []
            for ind in range(num_data):
                for k in keys:
                    if ind in indices[i]:
                        valid_data[k].append(data[k][ind])
                    else:
                        train_data[k].append(data[k][ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)
    else:
        num_data = len(data)
        for i in range(len(indices)):
            valid_data = []
            train_data = []
            for ind in range(num_data):
                if ind in indices[i]:
                    valid_data.append(data[ind])
                else:
                    train_data.append(data[ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds