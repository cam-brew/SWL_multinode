from re import L
import numpy as np
import time
from joblib import Parallel,delayed
from multiprocessing import Pool

import dask.array as da

from dask import delayed
from dask.diagnostics import ProgressBar
from scipy.ndimage import median_filter,binary_fill_holes, binary_closing,generate_binary_structure,convolve,gaussian_filter,binary_dilation
import skimage
from skimage.filters import threshold_otsu,threshold_triangle
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage import exposure
import matplotlib.pyplot as plt

"""
    Masking functions:
    - keep_largest_component(): deletes disconnected fields from mask
    - isolate_foreground(): applies Otsu threshold with binary closing and hole filling to create solid foreground mask
"""

def circular_mask(shape, radius_scale=0.95):
    d,h,w = shape
    cy,cx = h // 2, w // 2
    Y,X = np.ogrid[:h,:w]
    radius = min(h,w) * radius_scale / 2
    dist_from_cent = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return dist_from_cent <= radius

def sobel_edge_2d(tomo):
    Kx = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1,-2,-1]])
    
    Ky = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    
    Ix = convolve(tomo.astype(np.float32),Kx)
    Iy = convolve(tomo.astype(np.float32),Ky)

    edge_mag = np.sqrt(Ix**2 + Iy**2)

    return edge_mag

def hist_stretch(data,mask=None):
    if mask is not None:
        values = data[mask>0]
    else:
        values = data.ravel()
    p2,p98 = np.percentile(values,(2,98))
    return exposure.rescale_intensity(data,in_range=(p2,p98))

def keep_largest_component(mask,conn=2):
    labeled = label(mask, connectivity=conn)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def detect_spring(vol,intensity_thresh=0.95,std_thresh=0.25):
    n_slices = vol.shape[0]
    for i,slice in enumerate(vol.shape[0]):
        valid_px = slice[np.isfinite(slice)]
        vmax = np.percentile(valid_px,99)
        std_intensity = np.std(valid_px)
        if vmax > intensity_thresh or std_intensity > std_thresh:
            print(f'Spring detected in slice {i} (max={vmax:.2f}, std={std_thresh:.2f})')


def process_slice_np(slice,sigma_blur,sigma_mask):
    if np.all(np.isnan(slice)):
        return np.full_like(slice,np.nan), np.zeros_like(slice,dtype=np.uint8)

    valid = ~np.isnan(slice)
    mean = np.nanmean(slice)
    slice_fill = np.where(valid,slice,mean)

    blur = gaussian_filter(slice_fill,sigma=sigma_mask)

    try:
        t = threshold_triangle(blur[valid])
        mask = np.zeros_like(blur,dtype=np.uint8)
        mask[valid] = blur[valid] < t
    except ValueError:
        mask = np.zeros_like(blur,dtype=np.uint8)

    mask = binary_closing(mask)
    mask = binary_fill_holes(mask)
    mask = binary_dilation(mask)
    mask = keep_largest_component(mask)
    mask = mask.astype(np.uint8)
    
    if len(np.unique(mask[valid])) == 1:
        mask = np.zeros(mask.shape,dtype=np.int8)

    blur = gaussian_filter(slice_fill,sigma=sigma_blur)
    return blur,mask

def test_iso_np(vol,blur_kern_size=3,mask_kern_size=30):
    
    with Pool() as pool:
        results = pool.starmap(process_slice_np, [(vol[z],blur_kern_size,mask_kern_size) for z in range(vol.shape[0])])
    
    blur,mask = zip(*results)
    blur = np.stack(blur,axis=0)
    mask = np.stack(mask,axis=0)
    
    plt.imshow(mask[mask.shape[0] // 2])
    plt.show()
    # results = Parallel(n_jobs=1)(delayed(process_slice_np)(vol[z],blur_kern_size,mask_kern_size) for z in range(n_slices))
    # # blur,mask = zip(*results)
    # # blur = np.stack(blur,axis=0)
    # mask = np.stack(results,axis=0)
    return blur,mask
        
def rescale(vol):
    valid_mask = ~np.isnan(vol) # mask valid pixels
    
    # Set upper and lower clipping lims
    vmin = np.nanpercentile(vol,2)
    vmax = np.nanpercentile(vol,98)
    vol_clip = np.clip(vol,vmin,vmax)
    
    # Make vol [0,1]
    vol = (vol_clip - vmin) / (vmin - vmax)
    vol = vol.astype(np.float32)
    
    flat_valid = vol[valid_mask] # set valid mask
    vol_he = np.full_like(vol,np.nan) # initially empty set
    
    # only equalize locations within circular mask
    vol_he[valid_mask] = exposure.equalize_hist(flat_valid)
    
    return vol_he

def main():
    import tifffile
    import matplotlib.pyplot as plt
    from get_io import read_tomos_dask
    from monitor_performance import animate_stack
    from pathlib import Path

    tomo_dir = 'sample_slices_01/'
    f_names = sorted([f for f in Path(tomo_dir).iterdir() if f.suffix.lower() in ['.tif','.tiff'] and not '._' in f.name])

    print(f'Files: {len(f_names)} begining with {f_names[0]}')
    raw_tomos = read_tomos_dask(f_names,cores=None)
    raw_tomos = raw_tomos.compute()
    # Circular masking
    tomos = np.where(circular_mask(raw_tomos.shape,radius_scale=0.98), raw_tomos, np.nan)
    # Histogram Equialization
    tomos_he = rescale(tomos)
    
    blur,mask = test_iso_np(tomos_he,blur_kern_size=5,mask_kern_size=50)
    
    
    blur[mask != 1] = np.nan
    
    
    for i in range(len(tomos_he)):
        
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(raw_tomos[i],cmap='gray')
        ax[1].imshow(mask[i],cmap='gray')
        ax[2].imshow(blur[i],cmap='gray')
        plt.show()
    

if __name__ == '__main__':
    main()
