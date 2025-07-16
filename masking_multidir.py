from re import L
import numpy as np
import time


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


def test_iso_dask(vol, blur_kern_size = 3, mask_kern_size = 30):
    # Make sure input is Dask array with correct chunking
    mean = np.nanmean(vol)
    
    if not isinstance(vol, da.Array):
        vol_dask = da.from_array(vol, chunks=(1, vol.shape[1], vol.shape[2]))
    else:
        vol_dask = vol.rechunk((1, vol.shape[1], vol.shape[2]))

    print(f'Masking and filling')
    valid_mask = ~da.isnan(vol_dask)
    vol_filled = da.where(valid_mask,vol_dask,mean)

    # Compute global mean of valid voxels and fill NaNs

    # mean_val = da.nanmean(vol_dask).compute()

    print(f'Processing each slice')

    # --- Define a slice processing function ---
    
    def process_blur(slice2d,sigma):
        slice2d = slice2d[0]
        blurred = gaussian_filter(slice2d,sigma=sigma)
        return blurred[np.newaxis,:,:]
    
    
    def process_mask(slice2d, mask2d, mask_kern):
        
        # Change shape from (1,:,:) to (:,:)
        slice2d = slice2d[0]
        mask2d = mask2d[0]
        
        binary = np.zeros(slice2d.shape,dtype=np.uint8)
        blurred = gaussian_filter(slice2d, sigma=mask_kern)
        
        # Thresholding
        t = threshold_triangle(blurred[mask2d])
        binary[mask2d] = blurred[mask2d] < t
        
        # Close interior holes
        binary = binary_closing(binary)
        binary = binary_fill_holes(binary)
        binary = binary_dilation(binary,iterations=3)
        
        # Keep largest component function
        binary = label(binary, connectivity=2)
        sizes = np.bincount(binary.ravel())
        sizes[0] = 0
        binary == sizes.argmax()
        
        # Make sure array is not blank
        if np.unique(binary[mask2d]).size == 1:
            binary[:] = 0
            
        # Perform smaller gaussian blur to return
        return binary[np.newaxis,:,:]
    
    # --- Wrap for map_blocks ---

    # Apply processing functions via map_blocks
    blurred_dask = da.map_blocks(
        process_blur, vol_filled, blur_kern_size,
        dtype=np.float32,
        chunks=(1,vol.shape[1],vol.shape[2])
    )
    
    mask_dask = da.map_blocks(
        process_mask, vol_filled, valid_mask, mask_kern_size,
        dtype = np.int8,
        chunks = (1,vol.shape[1],vol.shape[2])
    )
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
    raw_tomos = read_tomos_dask(f_names,cores=None)
    raw_tomos = raw_tomos.compute()
    # Circular masking
    tomos = np.where(circular_mask(raw_tomos.shape,radius_scale=0.98), raw_tomos, np.nan)
    # Histogram Equialization
    tomos_he = rescale(tomos)
    
    blur,mask = test_iso_dask(tomos_he,blur_kern_size=5,mask_kern_size=50)
    
    blur_np = blur.compute()
    mask_np = mask.compute()
    
    tomos[mask_np != 1] = np.nan
    tomos[mask_np == 1] = blur_np[mask_np == 1]
    
    
    for i in range(len(blur)):
        
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(raw_tomos[i],cmap='gray')
        ax[1].imshow(mask_np[i],cmap='gray')
        ax[2].imshow(tomos[i],cmap='gray')
        plt.show()
    

if __name__ == '__main__':
    main()