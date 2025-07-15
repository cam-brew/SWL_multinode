import numpy as np
import psutil
import dask.array as da


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
                      max_voxels=750_000):
    """
    Dask-optimized version of GMM segmentation
    """
    # Flatten vals
    if isinstance(tomo_stack,np.ndarray):
        tomo_stack = da.from_array(tomo_stack,chunks=(1,tomo_stack.shape[1],tomo_stack.shape[2]))
    elif not isinstance(tomo_stack,da.Array):
        raise TypeError('Tomo stack must be Dask or Numpy')
    
    if isinstance(mask_stack,np.ndarray):
        mask_stack = da.from_array(mask_stack,chunks=(1,mask_stack.shape[1],mask_stack.shape[2]))
    elif not isinstance(mask_stack,da.Array):
        raise TypeError('Mask stack must be Dask or Numpy')
    
    valid_voxels = tomo_stack[mask_stack.astype(bool)]
    
    N = valid_voxels.size

    # Fit GMM on random subsample for saved mem
    if N > max_voxels:
        idx = np.random.choice(N, size=max_voxels, replace=False)
        sample = valid_voxels[idx].compute().reshape(-1,1)
    else:
        sample = valid_voxels.compute().reshape(-1,1)
        
    print('Creating GMM')
    gmm = gaussian_mix_init(n_classes)
    gmm.fit(sample)

    print('Creating dask array to write inside...')
    full_vals = valid_voxels.compute().reshape(-1,1)
    
    print('Predicting values...')
    predicted = gmm.predict(full_vals)
    
    # Step 5: Sort by mean intensities
    print('Remapping and sorting...')
    sorted_ind = np.argsort(gmm.means_.flatten())
    remap = np.zeros_like(sorted_ind)
    remap[sorted_ind] = np.arange(n_classes)
    sorted_labels = remap[predicted]

    print('Rebuild label volume...')
    label_volume = np.full(tomo_stack.shape,-1,dtype=np.int8)
    flat_mask = mask_stack.astype(bool).ravel()
    label_volume.ravel()[flat_mask] = predicted
    
    print(f'Label vol complete')
    return label_volume,gmm