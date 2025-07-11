import pickle
import multiprocessing
import numpy as np
import time
import os
import psutil
import dask.array as da
import dask

from concurrent.futures import ProcessPoolExecutor
from monitor_performance import setup_logger
from sklearn.mixture import GaussianMixture


"""
    Masking and classification definitions:
    - _predict_chunk(): private fnc only to be called through pickling by subsequent fncs
    - prob_parallel(): parallel calling of gaussian mixture probability prediction; appropriate chunk_size is essential to performance
    - gaussian_mix_init(): initialization of GaussianMixture() with common seed value for reproducibility (fnc from sklearn.mixture)
    - gaussian_mix(): general GMM function to be called by main()
"""

###### Dask version implementation #######

def gaussian_mix_init(n, covar_type='full', seed=0):
    return GaussianMixture(n_components=n, covariance_type=covar_type, random_state=seed)

def gaussian_mix_dask(tomo_stack, mask_stack, n_classes=2, confidence_threshold=0.9,
                      max_voxels=750_000, chunk_size=50_000):
    """
    Dask-optimized version of GMM segmentation
    """
    # Flatten vals
    print(f'Tomo shape: {tomo_stack.shape}')
    print(f'Mask shape: {mask_stack.shape}')
    masked_vals = np.where(mask_stack.astype(bool),tomo_stack,np.nan)
    values = masked_vals[~np.isnan(masked_vals)].reshape(-1,1)
    N = values.shape[0]

    # Fit GMM on random subsample for saved mem
    if N > max_voxels:
        idx = np.random.choice(N, size=max_voxels, replace=False)
        vals_fit = values[idx]
    else:
        vals_fit = values

    gmm = gaussian_mix_init(n_classes)
    gmm.fit(vals_fit)

    print('Creating dask array to write inside...')
    # Create a dask array from the full input values
    dask_vals = da.from_array(values, chunks=(chunk_size, 1))

    # Use map_blocks to apply predict_proba in parallel
    def predict_block(block):
        return gmm.predict(block)

    
    print(f'Mapping blocks...')
    dask_labels = dask_vals.map_blocks(predict_block,dtype=np.int8,drop_axis=1)
    
    print(f'Computing labels with {psutil.cpu_count(logical=False)} cores')
    labels = dask_labels.compute(num_worker=psutil.cpu_count(logical=False))
    
    # Step 5: Sort by mean intensities
    print('Remapping and sorting...')
    sorted_ind = np.argsort(gmm.means_.flatten())
    remap = np.zeros_like(sorted_ind, dtype=np.int8)
    remap[sorted_ind] = np.arange(n_classes)
    sorted_labels = remap[labels]

    print('Inserting into full volume...')
    # Step 6: Insert into volume
    gmm_labels = np.full(tomo_stack.shape, -1, dtype=np.int8)
    gmm_labels[mask_stack] = sorted_labels

    return gmm_labels, gmm