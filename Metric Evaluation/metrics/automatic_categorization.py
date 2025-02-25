import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(42)

def categorize_k(train_var,max_K=10,cut=0.85) :
    kk=max_K-2
    
    # K finder
    for i in range(kk):
        k_i=i+3
        Km=KMeans(n_clusters=k_i)
        Km.fit(pd.DataFrame(train_var))
        wss_new=Km.inertia_
        
        if i>0:
            
            if wss_old == 0:
                
                kk=k_i
                break
            
            else: 
                wss_ratio=wss_new/wss_old
                if wss_ratio>=cut:
                    kk=k_i
                    break
        
        wss_old=wss_new
        
    return(kk)

def categorization(data_orig, data_syn, conti_var, K_list):

    data_cat_orig=data_orig.copy()
    data_cat_syn=data_syn.copy()
    data_clu_or=data_orig.copy()

    # K-means -> categorization
    for i in range(len(conti_var)) :
        # K-means(original data) -> cluster -> categorization of original_data and synthetic_data

        ind_not_na=(data_orig[conti_var[i]].isna()==False)

        Km=KMeans(n_clusters=K_list[i])
        Km.fit(pd.DataFrame(data_orig[conti_var[i]][ind_not_na]))

        data_clu_or[conti_var[i]][ind_not_na]=(Km.labels_+1)

        clu_d=pd.DataFrame(pd.concat([data_clu_or[conti_var[i]],data_orig[conti_var[i]]],axis=1)).dropna()
        clu_d.columns=["clu","value"]
        clu_d=clu_d.groupby(['clu'],as_index=False).max().sort_values(by='value')
        clu_value=list(clu_d.value)

        # categorization
        data_cat_orig[conti_var[i]]=1
        data_cat_syn[conti_var[i]]=1

        for n in range(len(clu_value)-1):
            data_cat_orig[conti_var[i]]=data_cat_orig[conti_var[i]]+(data_orig[conti_var[i]]>clu_value[n])*1
            data_cat_syn[conti_var[i]]=data_cat_syn[conti_var[i]]+(data_syn[conti_var[i]]>clu_value[n])*1

        # NA인 데이터는 0으로 categorization
        if(data_orig[conti_var[i]].isna().sum()>0):
            data_cat_orig[conti_var[i]][data_orig[conti_var[i]].isna()]=0

        if(data_syn[conti_var[i]].isna().sum()>0):
            data_cat_syn[conti_var[i]][data_syn[conti_var[i]].isna()]=0
        
    return data_cat_orig, data_cat_syn


def categorize_df_kmeans(data_orig, data_syn, conti_var, auto_Kmeans = True, max_K = 10,cut = 0.85, K_list=None):
    
    data_orig = data_orig.copy()
    data_syn = data_syn.copy()
    
    if len(conti_var) == 0:

        pass

    elif auto_Kmeans == True:

        K_list = []

        for i in range(len(conti_var)):

            ind_not_na=(data_orig[conti_var[i]].isna()==False)
            K_list.append(categorize_k(data_orig[conti_var[i]][ind_not_na],max_K,cut))

        data_orig, data_syn = categorization(data_orig, data_syn, conti_var, K_list)

    else:

        data_orig, data_syn = categorization(data_orig, data_syn, conti_var, K_list)
        
    # treat NA as category level

    data_orig.fillna('NA', inplace = True)
    data_syn.fillna('NA', inplace = True)
    
    # set every value to str
    
    data_orig = data_orig.astype('str')
    data_syn = data_syn.astype('str')
    

    return data_orig, data_syn