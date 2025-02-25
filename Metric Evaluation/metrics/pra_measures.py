import pandas as pd
from tqdm import tqdm
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from sklearn.utils import shuffle

from representations.OneClass import * 
from evaluation import *

import warnings


def compute_metrics(real_data,synthetic_data, seed = 42, rep_dim = None):
    
    torch.manual_seed(seed)

    warnings.filterwarnings(action="ignore")
    
    
    # dummification for categorical data
    
    data = pd.concat([real_data,synthetic_data],axis=0)
    data = pd.get_dummies(data,drop_first=True)

    
    
    X = np.array(data.iloc[:real_data.shape[0],:])
    Y = np.array(data.iloc[real_data.shape[0]:,:])

    results = dict()
    
    if rep_dim == None:
        rep_dim = real_data.shape[1]
    
    params = dict({"rep_dim": rep_dim, 
                    "num_layers": 2, 
                    "num_hidden": 200, 
                    "activation": "ReLU",
                    "dropout_prob": 0.5, 
                    "dropout_active": False,
                    "train_prop" : 1,
                    "epochs" : 100,
                    "warm_up_epochs" : 10,
                    "lr" : 1e-3,
                    "weight_decay" : 1e-2,
                    "LossFn": "SoftBoundary"})   

    hyperparams = dict({"Radius": 1, "nu": 1e-2})

    params["input_dim"] = X.shape[1]       
    hyperparams["center"] = torch.ones(params['rep_dim'])
    
    # embedding of real_data

    model_real = OneClassLayer(params=params, hyperparams=hyperparams)
      
    model_real.fit(X,verbosity=False)

    X_out_real = model_real(torch.tensor(X).float()).float().detach().numpy()
    Y_out_real = model_real(torch.tensor(Y).float()).float().detach().numpy()
    
    alphas, alpha_precision_curve, Delta_precision_alpha, authenticity = compute_alpha_precision(X_out_real, Y_out_real, model_real.c)

    # embedding of synthetic_data

    model_synth = OneClassLayer(params=params, hyperparams=hyperparams)

    model_synth.fit(Y, verbosity = False)

    X_out_synth = model_synth(torch.tensor(X).float()).float().detach().numpy()
    Y_out_synth = model_synth(torch.tensor(Y).float()).float().detach().numpy()
    
    betas, beta_coverage_curve, Delta_coverage_beta, _ = compute_alpha_precision(Y_out_synth, X_out_synth, model_synth.c, compute_authen = False)

    results['Dpa'] = Delta_precision_alpha
    results['apc'] = alpha_precision_curve
    results['Dcb'] = Delta_coverage_beta
    results['bcc'] = beta_coverage_curve
    results['mean_aut'] = authenticity


    return results, model_real, model_synth


def plot_pr_curve(alpha_precision_curve, beta_coverage_curve, authenticity):
    
    alphas = np.linspace(0,1,30)

    plt.plot(alphas,alpha_precision_curve, color = 'blue', label = 'alpha-precision')
    plt.plot(alphas, beta_coverage_curve, color = 'purple', label = 'beta-recall')
    plt.plot([0, 1], [0, 1], linestyle='--', lw= 0.5, color='r')
    title = 'alpha-Precision and beta-Recall curve \n (Authenticity = ' + str(round(authenticity,3)) + ')'
    plt.title(title)
    plt.legend()
    plt.show()
    
    return plt
