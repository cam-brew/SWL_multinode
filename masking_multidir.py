from re import L
import numpy as np
import time


import dask.array as da

from dask import delayed
from dask.diagnostics import ProgressBar
from scipy.ndimage import median_filter,binary_fill_holes, binary_closing,generate_binary_structure,convolve,gaussian_filter
import skimage
from skimage.filters import threshold_otsu,threshold_triangle
from skimage.measure import label
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

def keep_largest_component(mask):
    labeled = label(mask, connectivity=2)
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


def test_iso_dask(vol, blur_kern_size = 3, mask_kern_size = 30):
    # Make sure input is Dask array with correct chunking
    if not isinstance(vol, da.Array):
        vol_dask = da.from_array(vol, chunks=(1, vol.shape[1], vol.shape[2]))
    else:
        vol_dask = vol.rechunk((1, vol.shape[1], vol.shape[2]))

    
    valid_mask = ~da.isnan(vol_dask)
    print(f'Valid mask')

    # Compute global mean of valid voxels and fill NaNs
    mean_val = vol_dask[valid_mask].mean().compute()
    vol_filled = da.where(valid_mask, vol_dask, mean_val)
    print(f'Vol filled')

    # --- Define a slice processing function ---
    def process_slice(slice2d, mask2d, kern):
        # Called per (1, y, x) block
        slice2d = slice2d[0]
        mask2d = mask2d[0]

        if not np.any(mask2d):
            
            binary = np.zeros(slice2d,dtype=np.uint8)
            blurred = gaussian_filter(slice2d,sigma=(kern,kern))
            return blurred[np.newaxis, :, :], binary[np.newaxis, :, :]
        
        
        blurred = gaussian_filter(slice2d, sigma=(kern, kern))
        t = threshold_otsu(blurred[mask2d])
        binary = np.zeros(slice2d.shape,dtype=np.uint8)
        
        binary[mask2d] = (blurred[mask2d] < t)
        
        binary = binary_closing(binary)
        binary = binary_fill_holes(binary)
        if np.unique(binary[mask2d]).size == 1:
            
            binary = np.zeros(slice2d.shape,dtype=np.uint8)
        # print(f'Unique elements {np.unique(binary).size}')
        # if np.unique(binary).size == 1:
        #     print(f'Empty slice detected')
        #     binary = np.zeros(binary.shape,dtype=np.uint8)
        return blurred[np.newaxis, :, :], binary[np.newaxis, :, :]

    # --- Wrap for map_blocks ---
    def blur_func(block, mask_block,blur_k):
        blur, _ = process_slice(block, mask_block, blur_k)
        return blur

    def mask_func(block, mask_block,mask_k):
        _, mask = process_slice(block, mask_block, mask_k)
        return mask

    print('Setting blur dask')
    # Apply processing functions via map_blocks
    blurred_dask = da.map_blocks(
        blur_func, vol_filled, valid_mask, blur_kern_size,
        dtype=np.float32,
        chunks=vol_filled.chunks
    )

    print('Setting mask dask')
    mask_dask = da.map_blocks(
        mask_func, vol_filled, valid_mask, mask_kern_size,
        dtype=np.uint8,
        chunks=vol_filled.chunks
    )

    # Restore NaNs to blurred image
    print(f'Masking blur')
    blurred_dask = da.where(valid_mask, blurred_dask, np.nan)

    return blurred_dask, mask_dask


        
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
    tomos = read_tomos_dask(f_names,cores=None)
    tomos = tomos.compute()
    
    tomos = np.where(circular_mask(tomos.shape,radius_scale=0.98), tomos, np.nan)
    
    tomos_he = rescale(tomos)
    
    
    
    blur,mask = test_iso_dask(tomos_he,kern_size=35)
    blur_np = blur.compute()
    mask_np = mask.compute()
    

    tomos[mask_np != 1] = np.nan
    tomos[mask_np == 1] = tomos_he[mask_np == 1]
    
    for i in range(len(blur)):
        if i == 0:
            fig,ax = plt.subplots(1,3)
            ax[0].imshow(blur_np[i])
            ax[1].imshow(mask_np[i])
            ax[2].imshow(tomos[i])
            plt.show()
    

    # for i,slice in enumerate(tomos):
    #     fig,ax = plt.subplots(1,3)
    #     ax[0].imshow(slice,cmap='gray',vmin=slice.min(),vmax=slice.max())
    #     ax[1].imshow(binary[i],cmap='gray')
    #     ax[2].imshow(np.multiply(slice,binary[i]),cmap='gray')
    #     plt.title(f'Slice {475 + i}')
    #     plt.show()
if __name__ == '__main__':
    main()