import numpy as np
import time
import dask.array as da

from dask import delayed
from dask.diagnostics import ProgressBar
from scipy.ndimage import median_filter,binary_fill_holes, binary_closing,generate_binary_structure,convolve,gaussian_filter
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






@delayed
def process_slice(slice,idx):
    
    if slice.ndim == 3:
        slice.squeeze(axis=0)

    print(slice.shape)
    
    median = median_filter(slice,size=5)
    edge = sobel_edge_2d(slice)

    enhanced = edge * median
    print(f'Slice {idx*200} std: {np.std(enhanced)}')

    if np.allclose(enhanced,0) or np.std(enhanced) < 0.001:
        return np.zeros_like(slice,dtype=bool)
    
    try:
        mask = enhanced > threshold_otsu(enhanced)
        return mask
    except Exception:
        print(f'[Slice{idx}] Triangle threshold failed')
        return np.zeros_like(slice,dtype=bool)

@delayed
def threshold(slice):
    try:
        thresh = threshold_triangle(slice)
        return (slice > thresh).astype(bool)
    
    except:
        return np.zeros_like(slice,dtype=bool)



def iso_foreground_dask(stack):
    
    Z,Y,X = stack.shape

    masks_z = [da.from_delayed(threshold(stack[z]),shape=(Y,X),dtype=bool) for z in range(Z)]
    mask_z_stack = da.stack(masks_z,axis=0)

    mask_x = [da.from_delayed(threshold(stack[:,:,x]),shape=(Z,Y),dtype=bool) for x in range(X)]
    mask_x_stack = da.stack(mask_x,axis=2)

    print(f'Z mask shape: {mask_z_stack.shape}')
    print(f'X mask shape: {mask_x_stack}')
    combined_mask = da.logical_and(mask_z_stack,mask_x_stack)
    return combined_mask
    
def edge_enhance_3d(vol):
    blur = gaussian_filter(vol,sigma=3)
    edges = np.sqrt(np.sum(np.square(np.gradient(blur)),axis=0))
    enhanced = blur + edges

    t = threshold_otsu(enhanced)
    binary = enhanced > t
    return binary,enhanced

def main():
    import tifffile
    import matplotlib.pyplot as plt
    from get_io import read_tomos_dask
    from monitor_performance import animate_stack
    from pathlib import Path

    tomo_dir = '/data/visitor/me1663/id19/20240227/PROCESSED_DATA/Real_05_01/delta_beta_150/Reconstruction_16bit_dff_s32_v2/Real_05_01_0001_16bit_vol/'
    f_names = sorted([f for f in Path(tomo_dir).iterdir() if f.suffix.lower() in ['.tif','.tiff'] and not '._' in f.name])
    f_names = f_names[::200]

    print(f'Files: {len(f_names)} begining with {f_names[0]}')
    tomos = read_tomos_dask(f_names,cores=None)
    tomos = tomos - tomos[0]

    # binary = iso_foreground_dask(tomos)
    # binary = binary.compute()
    binary,enhanced  = edge_enhance_3d(tomos)

    animate_stack(enhanced,binary,np.multiply(tomos,binary))

    # for i,slice in enumerate(tomos):
    #     fig,ax = plt.subplots(1,3)
    #     ax[0].imshow(slice,cmap='gray',vmin=slice.min(),vmax=slice.max())
    #     ax[1].imshow(binary[i],cmap='gray')
    #     ax[2].imshow(np.multiply(slice,binary[i]),cmap='gray')
    #     plt.title(f'Slice {475 + i}')
    #     plt.show()
if __name__ == '__main__':
    main()