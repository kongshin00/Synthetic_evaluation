def CAP(orig_data,syn_data,attribute):
    
    import pandas as pd
    import numpy as np
    
    """attribute must be list when input is more than 1"""
    if orig_data.shape[0] == 1 or syn_data.shape[0] == 1:
        raise ValueError('cannot calculate the CAP with only one categorical variable')
    data1 = orig_data.copy()
    data2 = syn_data.copy()
    #fill None -> 'NA'
    data1[attribute] = data1[attribute].fillna('NA')
    data2[attribute] = data2[attribute].fillna('NA')
    #keys : data column without attribute 
    keys = data1.drop(columns = attribute).columns.tolist()

    #unique attribue with row numbers
    uniq_attribute = data1[attribute].drop_duplicates()
    uniq_attribute['attribute_num'] = [i for i in range(uniq_attribute.shape[0])]
    #add attribute number to raw data
    data1 = pd.merge(data1,uniq_attribute,how = 'left')
    data2 = pd.merge(data2,uniq_attribute,how = 'left')

    #summarise unique count
    data1_uniq_cnt = data1.groupby(keys + ['attribute_num']).size().reset_index().rename(columns = {0:'n1'})
    data2_uniq_cnt = data2.groupby(keys + ['attribute_num']).size().reset_index().rename(columns = {0:'n2'})

    #left join + add NA row value
    data_uniq_cnt = pd.merge(data1_uniq_cnt,data2_uniq_cnt,how = 'left')
    data_NA_row = data2_uniq_cnt.groupby(keys)['n2'].sum().reset_index().rename(columns = {'n2':'NA_ROW'})

    #Make CAP score dataframe
    CAP_score = data_uniq_cnt.fillna(0)
    #   calculate cap score
    CAP_score = CAP_score.groupby(keys).apply(lambda x: sum(x.n2 / sum(x.n1) * x.n1)).reset_index()
    CAP_score = CAP_score.rename(columns = {0:'CAP'})
    #add N.row
    CAP_score = pd.merge(CAP_score,
    data_uniq_cnt.fillna(0).groupby(keys).n1.sum().reset_index().rename(columns = {'n1':'N_ROW'}),
    how = 'left')
    #Add NA.row
    CAP_score = pd.merge(CAP_score,data_NA_row,how = 'left').fillna(0)
    #calculate total score(CAP(0), CAP(NA))
    return {'CAP_0' : sum(CAP_score['CAP']) / (sum(CAP_score['N_ROW'])+1e-8),
    'CAP_NA' : sum(CAP_score['CAP']) / (sum(CAP_score['NA_ROW'])+1e-8)}
