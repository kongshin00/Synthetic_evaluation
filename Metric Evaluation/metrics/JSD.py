import numpy as np
import pandas as pd

def column_prob_cont(orig, synt, nbins = 100):
    
    total = pd.concat([orig,synt])
    
    bins = np.linspace(total.min(),total.max(),nbins)
    
    bins[0] = -np.Inf
    
    b_total = pd.cut(total, bins=bins, labels=np.arange(1,nbins))
    
    b_orig = b_total.iloc[:len(orig)]
    b_synt = b_total.iloc[len(orig):]
    
    orig_prob = []
    synt_prob = []
    
    for level in np.sort(b_total.unique()):
        orig_prob.append((b_orig == level).sum() / len(b_orig))
        synt_prob.append((b_synt == level).sum() / len(b_synt))
    
    return np.array(orig_prob), np.array(synt_prob)
    
    

def column_prob_disc(orig,synt):
    total = pd.concat([orig,synt])
    
    orig_prob = []
    synt_prob = []
    
    for level in np.sort(total.unique()):
        orig_prob.append((orig == level).sum() / len(orig))
        synt_prob.append((synt == level).sum() / len(synt))
        
    return np.array(orig_prob), np.array(synt_prob)

def kl_divergence(p, q):
    
    import warnings
    
    warnings.filterwarnings("ignore")
    
    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))

def js_divergence(p,q):
    
    m = (p + q)/2
    
    return kl_divergence(p,m)/2 + kl_divergence(q,m)


# compute JSD

# original data and synthetic data are pd.DataFrames.

def JSD(original_data, synthetic_data, var_info):
    
    col_info = dict()
    
    col_info['cont'] = []
    col_info['disc'] = []
    
    for col in var_info['numeric']:
        
        if len(original_data[col].unique()) > 100:
            
            col_info['cont'].append(col)
        else:
            
            col_info['disc'].append(col)
    
    col_info['disc'].extend(var_info['categorical'])
    col_info['disc'].extend(var_info['ordinal'])
        
        
    
    # get prob for each column
    
    col_prob = dict()
   
    for col in col_info['cont']:
        col_prob[col] = column_prob_cont(original_data[col], synthetic_data[col])
    
    for col in col_info['disc']:
        col_prob[col] = column_prob_disc(original_data[col], synthetic_data[col])
        
         
    # get JS-divergence for each column
    
    col_jsd = dict()
    
    jsd_data = 0
    
    for col in col_prob.keys():
        
        jsd = js_divergence(col_prob[col][0],col_prob[col][1])
        
        col_jsd[col] = jsd
        
        jsd_data += jsd
    
    
    return jsd_data,  col_jsd # return JSD of data and column-wise JSD.