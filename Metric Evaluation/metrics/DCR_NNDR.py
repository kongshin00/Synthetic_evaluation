# compute DCR and NNDR 
# var_info is a dictionary which indicates the type [numeric, categorical, ordinal] of each column with its name


import numpy as np
import pandas as pd

def DCR_NNDR(orig_data, syn_data, var_info):
    
    original_data = orig_data.copy()
    synthetic_data = syn_data.copy()
    
    n = original_data.shape[0]
    m = synthetic_data.shape[0]
    
    original_data.loc[:,var_info['ordinal']] = original_data.loc[:,var_info['ordinal']].astype(int)
    synthetic_data.loc[:,var_info['ordinal']] = synthetic_data.loc[:,var_info['ordinal']].astype(int) 
    
    # compute continuous scaler
    
    orig_cont = np.zeros((n,2))
    synt_cont = np.zeros((m,2))
    
    cont_scale = np.ones((2))
    
    
    if len(var_info['numeric']) > 0:
    

        orig_cont = np.array(original_data.loc[:,var_info['numeric']])
        synt_cont = np.array(synthetic_data.loc[:,var_info['numeric']])

        max_min = (orig_cont.max(axis = 0) - synt_cont.min(axis = 0))**2
        min_max = (orig_cont.min(axis = 0) - synt_cont.max(axis = 0))**2

        cont_scale = np.array([max_min,min_max]).max(axis = 0)
        
    
    # compute ordinal scaler
    
    orig_ordi = np.zeros((n,2))
    synt_ordi = np.zeros((m,2))
    
    ordi_scale = np.ones((2))
        
    if len(var_info['ordinal']) > 0:
    

        orig_ordi = np.array(original_data.loc[:,var_info['ordinal']])
        synt_ordi = np.array(synthetic_data.loc[:,var_info['ordinal']])

        max_min = np.abs(orig_ordi.max(axis = 0) - synt_ordi.min(axis = 0))
        min_max = np.abs(orig_ordi.min(axis = 0) - synt_ordi.max(axis = 0))

        ordi_scale = np.array([max_min,min_max]).max(axis = 0)
    
    orig_cate = np.zeros((n,2))
    synt_cate = np.ones((m,2))
    
    if len(var_info['categorical']) > 0:

        orig_cate = np.array(original_data.loc[:,var_info['categorical']])
        synt_cate = np.array(synthetic_data.loc[:,var_info['categorical']])
        
    # compute distance for each synthetic data
    
    dist_mat = np.zeros(shape = (m,2))
    
    for i in range(m):
        
        # conti distance (squared Euclidean)
        
        cont_dist = (((orig_cont - synt_cont[i,:])**2) / cont_scale).sum(axis = 1)
        ordi_dist = ((np.abs(orig_ordi - synt_ordi[i,:])) / ordi_scale).sum(axis = 1)
        cate_dist = (orig_cate == synt_cate[i,:]).sum(axis = 1)
        
        total_dist = np.sort(cont_dist + ordi_dist + cate_dist)
        
        dist_mat[i,] = total_dist[[0,1]]
        
    
    DCR = dist_mat[:,0]
    NNDR = dist_mat[:,0] / dist_mat[:,1]
    NNDR[np.isnan(NNDR)] = 1

    return DCR, NNDR
        
    
    
    
    
    
    
    