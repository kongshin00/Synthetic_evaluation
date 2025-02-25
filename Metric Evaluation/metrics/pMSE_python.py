def pMSE_compare(original_data,synthetic_data,n_permu=50,tree_min_split=20,tree_min_leaf=5,tree_max_depth = 30) : 
    import warnings
    warnings.filterwarnings(action="ignore")
    import random
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn import tree
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import copy
    
    syn=copy.deepcopy(synthetic_data)
    org=copy.deepcopy(original_data)
    n_iter=n_permu
    n1=syn.shape[0]
    n2=org.shape[0]
    c=n1/(n1+n2)
    
    # response variable
    syn["t"]=1
    org["t"]=0
    
    data=pd.concat([syn,org],axis=0)
    data=pd.get_dummies(data,drop_first=True)
    X=data.drop(["t"],axis=1)
    y=data["t"]
    
    # model training
    model_logistic=LogisticRegression(penalty = 'none', max_iter = 200)
    model_tree=tree.DecisionTreeClassifier(min_samples_split = tree_min_split, min_samples_leaf = tree_min_leaf,
                                          max_depth=tree_max_depth)   
    
    model_logistic.fit(X,y)
    model_tree.fit(X,y)
    
    # pMSE
    prop_lr=model_logistic.predict_proba(X)[:,1]
    prop_tree=model_tree.predict_proba(X)[:,1]
    
    pMSE_LR=np.mean((prop_lr-c)**2)
    pMSE_tree=np.mean((prop_tree-c)**2)
    
    #################################################
    # permutation -> Null distribution
    permu_pmse_LR=[]
    permu_pmse_tree=[]
    for i in range(n_permu):
        per_i=random.sample(range(n1+n2),n1) #random samling -> imputation
        data["t"]=0
        data["t"].iloc[per_i,]=data["t"].iloc[per_i,]+1
        
        X_permu=data.drop(["t"],axis=1)
        y_permu=data["t"]
        
        model_logistic.fit(X_permu,y_permu)
        model_tree.fit(X_permu,y_permu)
        prop_lr_permu=model_logistic.predict_proba(X_permu)[:,1]
        prop_tree_permu=model_tree.predict_proba(X_permu)[:,1]
        
        permu_pmse_LR.append(np.mean((prop_lr_permu-c)**2))
        permu_pmse_tree.append(np.mean((prop_tree_permu-c)**2))
    
    # array
    permu_pmse_LR=np.array(permu_pmse_LR)
    permu_pmse_tree=np.array(permu_pmse_tree)
    
    #pMSE-ratio , standardized pMSE
    pMSE_ratio_LR=pMSE_LR/(permu_pmse_LR.mean())
    pMSE_ratio_tree=pMSE_tree/(permu_pmse_tree.mean())
    s_pMSE_LR=(pMSE_LR-permu_pmse_LR.mean())/permu_pmse_LR.std()
    s_pMSE_tree=(pMSE_tree-permu_pmse_tree.mean())/permu_pmse_tree.std()
    
    ##################
    ## print plot
    #p_LR=sns.distplot(permu_pmse_LR,kde=True)
    #p_LR.set_title("pMSE Null Distribution by Logistic Regression")
    #plt.axvline(x=pMSE_LR,c="red",label='pMSE')
    #plt.legend()
    #plt.show()
    
    #p_dt=sns.distplot(permu_pmse_tree,kde=True)
    #p_dt.set_title("pMSE Null Distribution by Decision Tree")
    #plt.axvline(x=pMSE_tree,c="red",label='pMSE')
    #plt.legend()
    #plt.show()
    
    pmse_mat=np.array([[pMSE_LR,pMSE_ratio_LR,s_pMSE_LR],[pMSE_tree,pMSE_ratio_tree,s_pMSE_tree]])
    pmse_mat=pd.DataFrame(pmse_mat)
    pmse_mat.columns=["pMSE","s_pMSE","stanrdized_pMSE"]
    pmse_mat=pmse_mat.rename(index={"0":"Logistic","1":"Tree"})
    

    return (pmse_mat,permu_pmse_LR,permu_pmse_tree)