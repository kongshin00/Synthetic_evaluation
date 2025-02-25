
# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
  
  ----------------------------------------- 
  Metrics implementation
  ----------------------------------------- 

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

import logging
import torch
import scipy

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
device = 'cpu' # matrices are too big for gpu


def compute_alpha_precision(real_data, synthetic_data, emb_center, compute_authen = True):
    

    emb_center = torch.tensor(emb_center, device=device)

    n_steps = 30
    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)
        
    
    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)
    
    alpha_precision_curve = []
    
    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))


    
    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

 
        alpha_precision_curve.append(alpha_precision)
    
    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    
    
    authenticity = None
    
    if compute_authen == True:
    
    
        nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
        real_to_real, _       = nbrs_real.kneighbors(real_data)

        nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(real_data) # correction
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(synthetic_data)

        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
        real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
        real_to_synth_args    = real_to_synth_args.squeeze()



        # See which one is bigger

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen.numpy())

    
    return alphas, alpha_precision_curve, Delta_precision_alpha, authenticity
