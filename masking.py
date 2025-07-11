import numpy as np
import time
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
    print(f'Edge shape: {edge_mag.shape} {edge_mag.dtype}')

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
    tomo_stack = (tomo_stack / 65535.0).astype(np.float32)
    tomo_stack = tomo_stack - tomo_stack[0]

    edge=np.zeros(tomo_stack.shape,dtype=np.float32)
    binary=np.zeros(tomo_stack.shape,dtype=bool)
    
    for i,slice in enumerate(tomo_stack):
        edge[i] = sobel_edge_2d(slice)
        gauss = gaussian_filter(slice,sigma=(45.,45.))
        enhanced = np.multiply(gauss,edge[i])
        
        # plt.imshow(enhanced)
        # plt.title(f'Slice {i} std: {np.std(enhanced)}')
        # plt.show()
    
        # skip flat slices
        
        if np.allclose(enhanced,0) or np.std(enhanced) < 0.00015:
            binary[i] = np.zeros_like(slice,dtype=bool)
            continue
        try:
            binary[i] = enhanced > threshold_triangle(enhanced)
        except:
            print(f'[Slice {i}] Triangle threshold failed -- filling with zeros')
            binary[i] = np.zeros_like(slice,dtype=bool)

        binary[i] = keep_largest_component(binary[i])


    return binary

def main():
    import tifffile
    import matplotlib.pyplot as plt
    from get_io import read_tomos_dask
    from monitor_performance import animate_stack
    from pathlib import Path

    path = ''
    tomo_dir = '/data/visitor/me1663/id19/20240227/PROCESSED_DATA/Real_05_01/delta_beta_150/Reconstruction_16bit_dff_s32_v2/Real_05_01_0001_16bit_vol/'
    f_names = sorted([f for f in Path(tomo_dir).iterdir() if f.suffix.lower() in ['.tif','.tiff'] and not '._' in f.name])
    f_names = f_names[475:490]

    print(f'Files: {len(f_names)} begining with {f_names[0]}')
    tomos = read_tomos_dask(f_names,cores=None)
    binary = isolate_foreground(tomos)

    animate_stack(tomos,binary,np.multiply(tomos,binary))

    for i,slice in enumerate(tomos):
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(slice,cmap='gray',vmin=slice.min(),vmax=slice.max())
        ax[1].imshow(binary[i],cmap='gray')
        ax[2].imshow(np.multiply(slice,binary[i]),cmap='gray')
        plt.title(f'Slice {475 + i}')
        plt.show()
if __name__ == '__main__':
    main()