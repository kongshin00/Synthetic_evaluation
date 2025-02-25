import pandas as pd
import numpy as np
import warnings
import time
import datetime
import os
import matplotlib.pyplot as plt
import recordlinkage

def record_linakge(orig_data, syn_data, var_info, threshold = None):
    
    
    
    # Settings
    
    block_column = tuple(var_info['categorical'] + var_info['ordinal'])
    
    
    indexer = recordlinkage.Index()
    indexer.block(block_column)

    comp = recordlinkage.Compare()

    for var in var_info['numeric']:
        
        scale = orig_data[var].std()

        comp.numeric(var, var, method="exp", offset=0, scale=scale)
    
    candidate_links = indexer.index(orig_data, syn_data)
    
    features = comp.compute(candidate_links, orig_data, syn_data)
    
    features_sum = np.sum(features, axis=1).tolist()
    
    if threshold == None:
        
        threshold = len(var_info['numeric'])/2
    
    return sum(1 for num in features_sum if num >= threshold)