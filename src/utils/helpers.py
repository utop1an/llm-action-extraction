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
