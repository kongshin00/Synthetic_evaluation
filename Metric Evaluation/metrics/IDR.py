# orig : original dataset
# syn : synthetic dataset
# sensitive_var : list of sensitive variables names
# sensitive_conti_k : list that returns k if the sensitive variable is continuous and 0 if it is nominal.

def Disclosure(orig, syn, sensitive_var, auto_Kmeans = True, max_K = 10,cut = 0.85, sensitive_conti_k = None):
    from sklearn.cluster import KMeans
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_numeric_dtype

    def KMEANS(orig, syn, sensitive_var, sensitive_conti_k):
        # categorization
        data_cat_orig = orig.copy()
        data_cat_syn = syn.copy()
        data_clu_or = orig.copy()

        # K-means -> categorization
        # K-means(original data) -> cluster -> categorization of original_data and synthetic_data
        
        ind_not_na=(orig[sensitive_var].isna()==False)
        
        Km = KMeans(n_clusters=sensitive_conti_k)
        Km.fit(pd.DataFrame(orig[sensitive_var][ind_not_na]))
        data_clu_or[sensitive_var][ind_not_na]=(Km.labels_+1)
        
        clu_d=pd.DataFrame(pd.concat([data_clu_or[sensitive_var],orig[sensitive_var]],axis=1)).dropna()
        clu_d.columns=["clu","value"]
        clu_d=clu_d.groupby(['clu'],as_index=False).max().sort_values(by='value')
        clu_value=list(clu_d.value)

        # categorization
        sensitive_vark = sensitive_var+"_k"
        data_cat_orig[sensitive_vark]=1
        data_cat_syn[sensitive_vark]=1
        
        for n in range(len(clu_value)-1):
            data_cat_orig[sensitive_vark]=data_cat_orig[sensitive_vark]+(orig[sensitive_var]>clu_value[n])*1
            data_cat_syn[sensitive_vark]=data_cat_syn[sensitive_vark]+(syn[sensitive_var]>clu_value[n])*1
        
        # NA인 데이터는 0으로 categorization
        if(orig[sensitive_var].isna().sum()>0):
            data_cat_orig[sensitive_var][orig[sensitive_var].isna()] = 0
        
        if(syn[sensitive_var].isna().sum()>0):
            data_cat_syn[sensitive_var][syn[sensitive_var].isna()] = 0
        
        return data_cat_orig, data_cat_syn
    
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
    
    orig = orig.reset_index(drop = True)
    syn = syn.reset_index(drop = True)
    
    var_name = np.array(orig.columns)
    quasi_var = np.setdiff1d(var_name, sensitive_var)

    # number of quausi-identifiers
    n_quasi = np.size(quasi_var)

    # number of sensitive variables
    n_sensitive = np.size(sensitive_var)
    
    if auto_Kmeans == False:
    
        n_conti = 0; n_nominal = 0

        for i in sensitive_conti_k:
            if i != 0:
                n_conti += 1
            else:
                n_nominal += 1

        if n_conti != 0:
            for i in range(n_sensitive):
                if sensitive_conti_k[i] != 0:
                    orig, syn = KMEANS(orig, syn, sensitive_var[i], sensitive_conti_k[i])
                    
    else:
        
        sensitive_conti_k = []
        
        for sen_var in sensitive_var:
            
            if is_numeric_dtype(orig[sen_var]):
                
                ind_not_na=(orig[sen_var].isna()==False)
                
                auto_k = categorize_k(orig[sen_var][ind_not_na], max_K = max_K, cut = cut)
                
                sensitive_conti_k.append(auto_k)
                
                orig, syn = KMEANS(orig, syn, sen_var, auto_k)
            
            else:
                
                sensitive_conti_k.append(0)

    result = 0

    for i in range(orig.shape[0]):

        for j in range(n_quasi):
            globals()['value%s' % str(j+1)] = orig.loc[i][quasi_var[j]]

        df1 = orig
        # calculate f
        for j in range(n_quasi):
            df1 = df1[df1[quasi_var[j]] == globals()['value%s' % str(j+1)]]

        f = df1.shape[0]

        # measuring identification risk (calculate I)
        df2 = syn
        for j in range(n_quasi):
            df2 = df2[df2[quasi_var[j]] == globals()['value%s' % str(j+1)]]

        num = df2.shape[0]
        index1 = df2.index

        if num != 0:
            I = 1
        elif num == 0:
            I = 0

        # learning something new (calculate R)
        if I == 0:
            R = 0
        elif I != 0:
            L = 0

            for j in range(n_sensitive):
                var = sensitive_var[j]

                if sensitive_conti_k[j] == 0:
                    sensitive_value = np.sort(orig[var].unique())
                elif sensitive_conti_k[j] != 0:
                    vark = var+"_k"
                    sensitive_value = np.sort(orig[vark].unique())

                p_j_list = []; d_j_list = []
                for k in sensitive_value:
                    if sensitive_conti_k[j] == 0:
                        prop = orig[orig[var] == k].shape[0]/orig.shape[0]
                        p_j_list.append(prop)
                        d_j_list.append(1-prop)
                    elif sensitive_conti_k[j] != 0:
                        prop = orig[orig[vark] == k].shape[0]/orig.shape[0]
                        p_j_list.append(prop)
                        d_j_list = p_j_list

                # choose random among matched synthetic records
                index2 = np.random.choice(index1,1)[0]
                Y_t = syn.iloc[index2][var]

                if sensitive_conti_k[j] == 0:
                    X_s = orig.loc[i][var]
                elif sensitive_conti_k[j] != 0:
                    X_s = orig.loc[i][vark]
                index = np.where(sensitive_value == X_s)[0][0]

                # for nominal variable
                if sensitive_conti_k[j] == 0:

                    if X_s == Y_t:
                        value = 1
                    elif X_s != Y_t:
                        value = 0

                    p_j = p_j_list[index]; d_j = d_j_list[index]
                    value1 = d_j * value
                    value2 = np.sqrt(p_j*(1-p_j))

                    if value1 > value2:
                        L = L+1
                    elif value1 <= value2:
                        L = L+0

                # for continuous variable    
                elif sensitive_conti_k[j] != 0:

                    d_j = d_j_list[index]
                    value = abs(X_s - Y_t)
                    x = orig[var]
                    MAD = np.median(np.absolute(x - np.median(x)))

                    value1 = d_j * value
                    value2 = 1.48*MAD

                    if value1 < value2:
                        L = L+1
                    elif value1 >= value2:
                        L = L+0

            if L >= 0.05*n_sensitive:
                R = 1
            elif L < 0.05*n_sensitive:
                R = 0

        # final calculation
        final = 1/f * I * R
        result += final

    result = result / orig.shape[0]
    return result
