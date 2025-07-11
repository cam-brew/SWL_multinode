import numpy as np
import time
import dask.array as da

from dask import delayed
from dask.diagnostics import ProgressBar
from scipy.ndimage import binary_fill_holes, binary_closing,generate_binary_structure,convolve,gaussian_filter
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


def isolate_foreground(tomo_stack):
    
    # tomo_stack = (tomo_stack / 65535.0).astype(np.float32)
    tomo_stack = tomo_stack - tomo_stack[0]


    edge=np.zeros(tomo_stack.shape,dtype=np.float32)
    binary=np.zeros(tomo_stack.shape,dtype=bool)
    
    for i,slice in enumerate(tomo_stack):
        edge[i] = sobel_edge_2d(slice)
        gauss = gaussian_filter(slice,sigma=(45.,45.))
        enhanced = np.multiply(gauss,edge[i])
        print(f'Slice {i+475} std: {np.std(enhanced)}')
        # plt.imshow(enhanced)
        # plt.title(f'Slice {i} std: {np.std(enhanced)}')
        # plt.show()
    
        # skip flat slices
        
        if np.allclose(enhanced,0) or np.std(enhanced) < 0.001:
            binary[i] = np.zeros_like(slice,dtype=bool)
            continue
        try:
            binary[i] = enhanced > threshold_triangle(enhanced)
        except ValueError:
            print(f'[Slice {i}] Triangle threshold failed -- filling with zeros')
            binary[i] = np.zeros_like(slice,dtype=bool)

        binary[i] = keep_largest_component(binary[i])


    return binary

@delayed
def normalize_slice(slice,background):
    if slice.max() > slice.min():
        slice = (slice - slice.min()) / (slice.max() - slice.min())
        background = (background - slice.min()) / (slice.max() - slice.min())
        slice.astype(np.float32)
        background.astype(np.float32)
        return slice - background
    
    return np.zeros_like(slice,dtype=np.float32)

@delayed
def process_slice(slice,idx):
    
    if slice.ndim == 3:
        slice.squeeze(axis=0)

    print(slice.shape)
    edge = sobel_edge_2d(slice)
    gauss = gaussian_filter(slice,sigma=(45.,45.))
    enhanced = gauss * edge
    print(f'Slice {idx*200} std: {np.std(enhanced)}')

    if np.allclose(enhanced,0) or np.std(enhanced) < 0.001:
        return np.zeros_like(slice,dtype=bool)
    
    try:
        mask = enhanced > threshold_triangle(enhanced)
        return mask
    except Exception:
        print(f'[Slice{idx}] Triangle threshold failed')
        return np.zeros_like(slice,dtype=bool)
    

def iso_foreground_dask(tomo_stack):
    # delay_slice = tomo_stack.to_delayed().flatten()

    norm_func = [normalize_slice(d,tomo_stack[0]) for d in tomo_stack]
    norm_stack = da.stack([da.from_delayed(n,shape=tomo_stack.shape[1:],dtype=np.float32) for n in norm_func], axis=0)
    # for i in range(norm_stack):
    #     plt.imshow(norm_stack[i])
    #     plt.show()
    print(norm_stack.shape)
    # norm = norm_stack - norm_stack[0]
    
    binary_func = [process_slice(norm_stack[i],i) for i in range(len(norm_stack))]
    binary_stack = da.stack([da.from_delayed(b, shape=norm_stack.shape[1:], dtype=bool) for b in binary_func], axis=0)

    return binary_stack
    

def main():
    import tifffile
    import matplotlib.pyplot as plt
    from get_io import read_tomos_dask
    from monitor_performance import animate_stack
    from pathlib import Path

    path = ''
    tomo_dir = '/data/visitor/me1663/id19/20240227/PROCESSED_DATA/Real_05_01/delta_beta_150/Reconstruction_16bit_dff_s32_v2/Real_05_01_0001_16bit_vol/'
    f_names = sorted([f for f in Path(tomo_dir).iterdir() if f.suffix.lower() in ['.tif','.tiff'] and not '._' in f.name])
    f_names = f_names[::200]

    print(f'Files: {len(f_names)} begining with {f_names[0]}')
    tomos = read_tomos_dask(f_names,cores=None)
    
    binary = iso_foreground_dask(tomos)
    binary = binary.compute()

    animate_stack(tomos,binary,np.multiply(tomos,binary))

    # for i,slice in enumerate(tomos):
    #     fig,ax = plt.subplots(1,3)
    #     ax[0].imshow(slice,cmap='gray',vmin=slice.min(),vmax=slice.max())
    #     ax[1].imshow(binary[i],cmap='gray')
    #     ax[2].imshow(np.multiply(slice,binary[i]),cmap='gray')
    #     plt.title(f'Slice {475 + i}')
    #     plt.show()
if __name__ == '__main__':
    main()