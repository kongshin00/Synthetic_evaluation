import pandas as pd
import numpy as np

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def obtain_eta(data, type_dict):
    
    n_cat = len(type_dict['cat'])
    n_cont = len(type_dict['cont'])
    
    result = np.zeros(shape = (n_cat+n_cont,n_cat+n_cont))
    
    result = pd.DataFrame(result)
    
    result.index = type_dict['cat'] + type_dict['cont']
    result.columns = type_dict['cat'] + type_dict['cont']
    
    for cat in type_dict['cat']:
        for cont in type_dict['cont']:
            
            formula = cont+' ~ '+cat
            
            model = ols(formula, data).fit()
            
            aov = anova_lm(model)
            
            eta = np.sqrt(aov.iloc[0,1] / (aov.iloc[1,1] + aov.iloc[0,1]))
            
            result.loc[cat,cont] = eta
            result.loc[cont,cat] = eta
    
    return result

def obtain_spearman(data, type_dict):
    
    import copy
    
    spearman_df = copy.deepcopy(data.loc[:,type_dict['cont']+type_dict['ord']])
    spearman_df.loc[:,type_dict['cont']] = spearman_df.loc[:,type_dict['cont']].rank()
    return spearman_df.corr()

def obtain_cramer(data, type_dict):
    
    import copy
    
    import association_metrics as am
    
    cat_vars = copy.deepcopy(data.loc[:,type_dict['bin']+type_dict['cat']+type_dict['ord']])
    
    cat_vars = cat_vars.apply(lambda x: x.astype("category"))
    
    cramersv = am.CramersV(cat_vars)
    
    return cramersv.fit()

def pairwise_corr(data, type_dict):
    
    import copy
    
    data = copy.deepcopy(data)
     
    p = data.shape[1]
    
    final_corr = pd.DataFrame(np.zeros(shape = (p,p)))
    
    final_corr.index = data.columns
    final_corr.columns = data.columns
    
    # obtain pearson correlation, phi_coefficient, point_biserial, rank_point_biserial... 
    pearson_corr = data.loc[:,type_dict['cont']+type_dict['bin']+type_dict['ord']].corr()
    
    # obtain cramer_V
    cramer_corr = obtain_cramer(data, type_dict)
    
    # obtain eta coefficient
    eta_corr = obtain_eta(data, type_dict)
    # obtain spearman coefficient
    
    spearman_corr = obtain_spearman(data, type_dict)
    
    # reconstruct correlation matrix
    
    ## for continuous variables 

    # obtain pearson and point-biserial coefficient
    final_corr.loc[type_dict['cont'],type_dict['cont']+type_dict['bin']] = pearson_corr.loc[type_dict['cont'],type_dict['cont']+type_dict['bin']] 

    # obtain eta coefficient
    final_corr.loc[type_dict['cont'],type_dict['cat']] = eta_corr.loc[type_dict['cont'],type_dict['cat']]

    # obtain spearman coefficient
    final_corr.loc[type_dict['cont'], type_dict['ord']] = spearman_corr.loc[type_dict['cont'],type_dict['ord']]

    ## for binary variables

    # obtain phi coefficient, point-biserial, rank-point-biserial

    final_corr.loc[type_dict['bin'],type_dict['cont']+type_dict['bin']+type_dict['ord']] = pearson_corr.loc[type_dict['bin'],type_dict['cont']+type_dict['bin']+type_dict['ord']]

    # obtain cramer V

    final_corr.loc[type_dict['bin'],type_dict['cat']] = cramer_corr.loc[type_dict['bin'],type_dict['cat']]

    ## for categorical variables

    # obtain eta coefficient

    final_corr.loc[type_dict['cat'],type_dict['cont']] = eta_corr.loc[type_dict['cat'],type_dict['cont']]

    # obtain cramer V

    final_corr.loc[type_dict['cat'],type_dict['bin']+type_dict['cat']+type_dict['ord']] = cramer_corr.loc[type_dict['cat'],type_dict['bin']+type_dict['cat']+type_dict['ord']]


    ## for ordianl variables

    # obtain spearman coefficient

    final_corr.loc[type_dict['ord'],type_dict['cont']+type_dict['ord']] = spearman_corr.loc[type_dict['ord'],type_dict['cont']+type_dict['ord']]

    # obtain rank-biserial

    final_corr.loc[type_dict['ord'], type_dict['bin']] = pearson_corr.loc[type_dict['ord'], type_dict['bin']]

    # obtain cramer V

    final_corr.loc[type_dict['ord'], type_dict['cat']] = cramer_corr.loc[type_dict['ord'], type_dict['cat']]
    
    return final_corr


def compare_pairwise_corr(data_orig, data_syn, type_dict):
    
    # prerpocess data
    
    # encode binary to 0,1 if not
    
    import copy
    
    data_orig = copy.deepcopy(data_orig)
    data_syn = copy.deepcopy(data_syn)
    
    for bin_ in type_dict['bin']:
        
        if ~np.isin(data_orig[bin_].unique(), np.array([0,1])).all():
        
            unique = data_orig[bin_].unique()

            data_orig[bin_] = data_orig[bin_].map({unique[0]:0,unique[1]:1})
            data_syn[bin_] = data_syn[bin_].map({unique[0]:0,unique[1]:1})
    
    # set categorical variables to strㅇ
    
    data_orig.loc[:,type_dict['cat']] = data_orig.loc[:,type_dict['cat']].astype(str)
    data_syn.loc[:,type_dict['cat']] = data_syn.loc[:,type_dict['cat']].astype(str)
    
    # set the order of columns equal
    
    data_syn = data_syn.reindex(columns=data_orig.columns)
    
    # compute pairwise corr
    
    corr_orig = pairwise_corr(data_orig,type_dict)
    
    corr_syn = pairwise_corr(data_syn,type_dict)
    
    # compute Frobenius Norm of the difference
    
    return corr_orig, corr_syn, np.linalg.norm(np.array(corr_orig)-np.array(corr_syn))