import warnings
warnings.filterwarnings(action="ignore")
import random
import pandas as pd
import numpy as np
from automatic_categorization import *

def pop_uni(original_data,synthetic_data,var_info,auto_Kmeans = True, max_K = 10,cut = 0.85, K_list=None) : 
    from sklearn.cluster import KMeans
    import pandas as pd
    import numpy as np
    
    
    data_orig=original_data
    data_syn=synthetic_data[data_orig.columns]
    
    conti_var = var_info['numeric']
    
    ###########################################################################################
    # categorization
    
    if len(conti_var)==0 :
        df_or=data_orig
        df_syn=data_syn
        
    elif auto_Kmeans == True:
        
        K_list = []
        
        for i in range(len(conti_var)):
            
            ind_not_na=(data_orig[conti_var[i]].isna()==False)
            K_list.append(categorize_k(data_orig[conti_var[i]][ind_not_na],max_K,cut))
        
        df_or, df_syn = categorization(data_orig, data_syn, conti_var, K_list)
        
    else:
        
        df_or, df_syn = categorization(data_orig, data_syn, conti_var, K_list)
    
    ###########################################################################################
    # Population uniqueness

    no_dup_X_syn = df_syn.drop_duplicates()
    count_syn=[]
    for i in range(no_dup_X_syn.shape[0]):
        syn_i=df_syn[(df_syn==no_dup_X_syn.iloc[i,:]).sum(axis=1)==no_dup_X_syn.shape[1]]
        count_syn.append(syn_i.shape[0])

    uni_X_syn=no_dup_X_syn[np.array(count_syn)==1]

    # unique case of original data among unique synthetic data
    count_or=[]
    for i in range(uni_X_syn.shape[0]):
        count_orig_i=sum((df_or==uni_X_syn.iloc[i,:]).sum(axis=1)==uni_X_syn.shape[1])
        count_or.append(count_orig_i)

    population_uni=sum(np.array(count_or)==1)/(uni_X_syn.shape[0]+1e-8)
    
    
    return (population_uni)