from automatic_categorization import *

def CAP(original_data,synthetic_data, attribute, var_info, auto_Kmeans = True, max_K = 10,cut = 0.85, K_list=None) : 
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
    # CAP
   
    X_or=df_or.drop([attribute],axis=1)
    X_syn=df_syn.drop([attribute],axis=1)
    
    # list of unique values for each vavriable
    unique_X_or = X_or.drop_duplicates()

    # CAP
    na_count=0
    p_sum=0
    for i in range(len(unique_X_or)):
        target_list=unique_X_or.iloc[i,]
        target_or=df_or[((X_or==target_list).sum(axis=1)==X_or.shape[1])]
        
        target_syn=df_syn[((X_syn==target_list).sum(axis=1)==X_or.shape[1])]

        if target_syn.shape[0]==0 :
            na_count=na_count+ target_or.shape[0]
        else :
            attribute_matrix=np.zeros([2,target_or[attribute].nunique()])
            for t in range(len(target_or[attribute].unique())):
                uni_t=target_or[attribute].unique()
                attribute_matrix[0,t]=target_or[target_or[attribute]==uni_t[t]].shape[0]
                attribute_matrix[1,t]=target_syn[target_syn[attribute]==uni_t[t]].shape[0]/target_syn.shape[0]

            p_sum=p_sum+sum(attribute_matrix[0,]*attribute_matrix[1,])

    CAP_0=p_sum/data_orig.shape[0] # CAP_0 (consider case not in original as 0)
    if data_orig.shape[0]-na_count == 0:
        CAP_NA = None
    else:
        CAP_NA=p_sum/(data_orig.shape[0]-na_count) # CAP_NA (do not consider case not in original)
    
    
    return (CAP_0,CAP_NA)