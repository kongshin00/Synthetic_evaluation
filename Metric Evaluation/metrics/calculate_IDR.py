# IDR function
# Original data and synthetic data should only have categorical variables.

def calculate_IDR(orig_data, syn_data, sensitive):
    
    import pandas as pd
    import numpy as np
    import copy
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    if orig_data.shape[1] == 1:
        
        print('Can not calculate the IDR with only one categorical variable')
        
        return np.nan, np.nan
    
    else:
        
        orig_data = copy.deepcopy(orig_data)
        syn_data = copy.deepcopy(syn_data)
        
        # treat NA as category level
    
        orig_data.fillna('NA', inplace = True)
        syn_data.fillna('NA', inplace = True)

        quasi = list(set(orig_data.columns) - set(sensitive))

        # count f

        count = orig_data.groupby(quasi).size().reset_index(name='Count')
        orig_count = pd.merge(orig_data, count, how = 'left', on = quasi)

        # Find I

        orig_quasi = orig_data.loc[:,quasi]
        syn_quasi = syn_data.loc[:,quasi]

        unique_quasi = orig_quasi.drop_duplicates()

        unique_quasi['combined'] = unique_quasi.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        syn_quasi['combined'] = syn_quasi.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        unique_quasi['Ind']=unique_quasi.combined.isin(syn_quasi.combined)
        unique_quasi = unique_quasi.drop(['combined'],axis = 1)

        orig_Ind = pd.merge(orig_count, unique_quasi, how = 'left', on = quasi)

        # For each sensitive variables Find R

        inequality_count = np.zeros(orig_data.shape[0])

        n_sen = len(sensitive)

        for i in range(n_sen):

            sen_var = sensitive[i] 

            quasi_sen = quasi + [sen_var]

            sen_df = pd.merge(orig_data.loc[:,quasi_sen], syn_data.loc[:,quasi_sen], how = 'left', on = quasi).drop_duplicates()

            # find whether there is a match for sensitive variable

            sen_df['match'] = (sen_df.iloc[:,-1] == sen_df.iloc[:,-2])

            sen_match = sen_df.loc[sen_df['match'] == True,:]
            sen_match.drop([sensitive[i]+'_y'],axis = 1, inplace = True)
            sen_match.columns = quasi_sen + ['match']

            orig_sen = orig_data.loc[:,quasi_sen]
            orig_sen_match = pd.merge(orig_sen, sen_match, on = quasi_sen, how = 'left')
            orig_sen_match['match'] = orig_sen_match['match'].fillna(False)

            # find p for each sensitive variable

            p_df = pd.DataFrame(orig_sen_match[sen_var].value_counts() / len(orig_sen_match)).reset_index()
            p_df.columns = [sen_var,'p']

            orig_sen_match_p = pd.merge(orig_sen_match,p_df, on = sen_var, how = 'left')

            match = orig_sen_match_p['match']
            p = orig_sen_match_p['p']
            d = 1 - p

            # check the inequality

            inequality_count += np.array(d * match > np.sqrt(p*d))

        R = (inequality_count > n_sen*0.05)

        IDR = (1 / (orig_Ind['Count'] + 1e-8)) * orig_Ind['Ind'] * R

        mean_IDR = np.mean(IDR)
        
        # IDR: individual IDR for each observation in original data
        # mean_IDR: average IDR of whole data

        return IDR, mean_IDR