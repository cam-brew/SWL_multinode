import numpy as np
import time
from multiprocessing import Pool
import multiprocessing
from scipy.ndimage import binary_fill_holes,binary_closing,convolve,gaussian_filter,binary_dilation,binary_erosion,maximum_filter
from skimage.filters import threshold_otsu,threshold_triangle
from scipy.ndimage import label as label3d
from skimage.measure import label as label2d
from skimage.morphology import disk,remove_small_objects
from skimage import exposure
from segmentation import gaussian_mix_np

"""
    Masking functions:
    - keep_largest_component(): deletes disconnected fields from mask
    - isolate_foreground(): applies Otsu threshold with binary closing and hole filling to create solid foreground mask
"""

def circular_mask(shape, radius_scale=0.95):
    if len(shape) == 2:
        h,w = shape
    elif len(shape) == 3:
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

def keep_largest_component_2d(mask,conn=1):
    labeled = label2d(mask, connectivity=conn)
    if labeled.max() == 0:
        return mask
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


def keep_largest_component_3d(stack,id,conn=6):
    if stack.ndim != 3:
        raise ValueError(f'Expected 3D mask, got shape: {stack.shape}')
    if conn==6:
        conn=1
        # structure = np.zeros((3,3,3),dtype=np.int8)
        # structure[1,1,:] = 1
        # structure[1,:,1] = 1
        # structure[:,1,1] = 1
    elif conn==26:
        conn = 3
        # structure = np.ones((3,3,3),dtype=np.int8)
    else:
        raise ValueError('conn must be 6 or 26')
    mask = (stack == id)
    if not np.any(mask):
        print('Mask considered empty')
        return stack.copy()
    labeled_mask = label2d(mask,connectivity=conn)
    if labeled_mask.max() == 0:
        print('Detected no features')
        return stack.copy()

    comp_size = np.bincount(labeled_mask.ravel())
    comp_size[0] = 0
    largest_label = comp_size.argmax()
    result = stack.copy()

    result[mask] = stack.min()
    result[labeled_mask == largest_label] = id
    return result

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
    mask = keep_largest_component_2d(mask)
    mask = mask.astype(np.uint8)
    
    if len(np.unique(mask[valid])) == 1:
        mask = np.zeros(mask.shape,dtype=np.int8)

    blur = gaussian_filter(slice_fill,sigma=sigma_blur)
    return blur,mask

def gaussian_blur(slice,sigma):
    return gaussian_filter(slice,sigma=sigma)



def close_mask_2(mask,iters=30,valid=None,erode_frac=0.5):
    mask = binary_dilation(mask,iterations=iters)
    mask = binary_closing(mask,structure=disk(3))
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask,structure=None,iterations=int(erode_frac*iters))
    if np.any(valid) is not None and np.any(mask):
        unique_vals = np.unique(mask[valid])
        # print(f'Unique values: {unique_vals}')
        if len(unique_vals) == 1:
            # print(f'Detected blank slice')
            mask = np.zeros(mask.shape,dtype=bool)
    return mask.astype(bool)

def isolate_foreground_AAU_2(vol,blur_kern_size=3,mask_kern_size=30):

    vol_he = rescale(vol,clip=0.3)

    circ = np.where(circular_mask(vol.shape,radius_scale=0.90),vol,np.nan)
    valid = ~np.isnan(circ)

    print(f'Valid shape: {valid.shape}')
    # vol = np.nan_to_num(vol,nan = np.max(vol[valid]))


    with Pool() as pool:
        edge_first = pool.map(sobel_edge_2d,[(tomo) for tomo in vol_he])
        edge_first = np.stack(edge_first)
        edge_first = np.where(circular_mask(edge_first.shape,radius_scale=0.90),edge_first,0)
        edge_first = pool.starmap(gaussian_blur,[(tomo,blur_kern_size) for tomo in edge_first])
        edge_mask = np.zeros_like(edge_first,dtype=np.uint8)
        edge_first = np.stack(edge_first)

        print(f'Edge shape: {edge_first.shape}')
        t_edge = threshold_otsu(edge_first[valid].ravel())
        edge_mask[valid] = edge_first[valid] > t_edge

        blur = pool.starmap(gaussian_blur,[(tomo,mask_kern_size) for tomo in vol])
        blur = np.stack(blur)
        blur = np.where(circular_mask(vol.shape,radius_scale=0.90),blur,0)
        t_blur = threshold_otsu(blur[valid].ravel())
        blur_mask = np.zeros_like(blur,dtype=np.uint8)
        blur_mask[valid] = blur[valid] > t_blur

        out = np.logical_and(edge_mask,blur_mask,dtype=np.uint8)
        out = pool.starmap(remove_small_objects,[(out[i],300) for i in range(out.shape[0])])
        # out = keep_largest_component_3d(out,id=1,conn=6)
        out = pool.starmap(close_mask_2,[(out[i],100) for i in range(len(out))])
        # edge_mask = pool.starmap(remove_small_objects,([(edge_mask[i],50) for i in range(edge_mask.shape[0])]))
        # edge_mask = maximum_filter(edge_mask,size=3)
        # mask = pool.starmap(close_mask_2,([(edge_mask[i],100) for i in range(len(edge_mask))]))
        out = pool.starmap(binary_erosion,([(out[i],None,90) for i in range(len(out))]))


        # print(f'Performing GMM')   
        # out,_ = gaussian_mix_np(vol,np.stack(mask))
        # out = (out == 0).astype(np.uint8)
        # out = np.stack(out)
        # out = pool.starmap(gaussian_blur,[(out[i],3) for i in range(out.shape[0])])
        # out = np.stack(out)
        # t = threshold_otsu(out.ravel())
        
        # out = (out > t).astype(np.uint8)
        # out = keep_largest_component_3d(out,id=out.max(),conn=6)

        # out = pool.starmap(close_mask_2,[(out[i],10) for i in range(out.shape[0])])


    return np.stack(out)
    
def isolate_foreground_AAU(vol,blur_kern_size=3,mask_kern_size=30):
    
    valid = ~np.isnan(vol)
    vol_filled = np.nan_to_num(vol,nan=np.max(vol[valid]))

    with Pool() as pool:
        blur = pool.starmap(gaussian_blur,[(vol_filled[z],mask_kern_size) for z in range(vol.shape[0])])

        blur = np.stack(blur,axis=0)
        vals = blur[valid].ravel()
        print(f'Max: {vals.max()} | Min: {vals.min()}')
        t = threshold_otsu(blur[valid].ravel())
        mask = np.zeros_like(vol,dtype=np.int8)
        mask[valid] = blur[valid] < t

        mask = keep_largest_component_3d(mask,id=1)

    # with Pool() as pool:
        if blur_kern_size != None:
            blur = pool.starmap(gaussian_blur,[(vol_filled[z],blur_kern_size) for z in range(vol.shape[0])])
        
        mask_closed = pool.starmap(close_mask,[(mask[i],valid[i],35) for i in range(mask.shape[0])])

    if blur_kern_size != None:
        blur = np.stack(blur,axis=0)
    else:
        blur = vol
    mask = np.stack(mask_closed,axis=0)
    
    return blur,mask

def isolate_foreground_COM(vol,blur_kern_size=3,mask_kern_size=30):
    
    valid = ~np.isnan(vol)
    vol_filled = np.nan_to_num(vol,nan=np.max(vol[valid]))


    with Pool() as pool:
        print('Blurring')
        blur = pool.starmap(gaussian_blur,[(vol_filled[z],mask_kern_size) for z in range(vol.shape[0])])
        print('Retrieving edges')
        edges = pool.map(sobel_edge_2d,blur)
        print('Selecting edges')
        edges_otsu = pool.map(threshold_otsu,edges)
        print('Closing interior')
        mask = pool.starmap(close_mask,[(edges_otsu[z],valid[z],1) for z in range(len(edges_otsu))])
        print('Closed interior')
    # vals = blur[valid].ravel()
    # print(f'Max: {vals.max()} | Min: {vals.min()}')
    # t = threshold_otsu(blur[valid].ravel())
    # mask = np.zeros_like(vol,dtype=np.int8)
    # mask[valid] = blur[valid] < t

    # mask = keep_largest_component_3d(mask,id=1)

    # with Pool() as pool:
    #     if blur_kern_size != None:
    #         blur = pool.starmap(gaussian_blur,[(vol_filled[z],blur_kern_size) for z in range(vol.shape[0])])
    #     mask_closed = pool.starmap(close_mask,[(mask[i],valid[i],5) for i in range(mask.shape[0])])

        if blur_kern_size in (0,1,None):
            print('No gaussian applied to scan')
            blur = vol
        else:
            print('Final blur')
            blur = pool.starmap(gaussian_blur,[(vol_filled[z],blur_kern_size) for z in range(vol.shape[0])])
            blur = np.stack(blur,axis=0)

    mask = np.stack(mask,axis=0)
    print(f'Number of elements in mask: {len(np.unique(mask))}')
    
    return blur,mask
        
def rescale(vol,clip=2, mask=None,hist_eq=False):

    if mask is None:
        mask = ~np.isnan(vol)
    
    valid_vals = vol[mask]
    if valid_vals.size == 0:
        print(f'No values to mask')
        return vol
    
    # valid_mask = ~np.isnan(vol) # mask valid pixels
    
    # Set upper and lower clipping lims
    vmin = np.nanpercentile(valid_vals,clip)
    vmax = np.nanpercentile(valid_vals,100-clip)
    vol_clip = np.clip(vol,vmin,vmax)
    
    # Make vol [0,1]
    vol_norm = (vol_clip - vmin) / (vmax - vmin)
    vol_norm = vol_norm.astype(np.float32)
    if hist_eq == True:
        print(f'Performing histogram equalization')
        flat_valid = vol[mask] # set valid mask
        vol_he = np.full_like(vol,np.nan) # initially empty set
        
        #only equalize locations within circular mask
        vol_he[mask] = exposure.equalize_hist(flat_valid)
        return vol_he
    
    return vol_norm


def masking_blur(stack,sigma,mask=None,keep_largest_comp=False):
    if mask is None:
        mask = np.zeros(stack.shape[1:],dtype=np.uint8)

    circ = np.where(circular_mask(stack.shape,radius_scale=0.90),stack,np.nan)
    valid = ~np.isnan(circ)

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        start = time.time()
        blur = pool.starmap(gaussian_blur,[(tomo,sigma) for tomo in stack])
        print(f'Blur time: {time.time() - start}')
        start = time.time()
        blur = rescale(np.stack(blur),clip=0.3)
        print(f'Normalize: {time.time() - start}')
        start = time.time()
        blur = np.where(mask,blur,0)
        out = np.zeros_like(stack,dtype=np.uint8)

        t = threshold_otsu(blur[valid].ravel())
        out[valid] = blur[valid] > t
        print(f'Threshold: {time.time() - start}')
        start = time.time()
        if keep_largest_comp == True:
            out = pool.map(keep_largest_component_2d,[tomo for tomo in out])
        print(f'Keep component: {time.time() - start}')
        start = time.time()
        out = pool.starmap(close_mask_2,[(tomo,10) for tomo in out])
        print(f'Closing sequence: {time.time() - start}')
        start = time.time()
        out = pool.starmap(binary_erosion,[(tomo,None,10) for tomo in out])
        print(f'Erosion: {time.time() - start}')

    return np.stack(out)


def masking_edge(stack,sigma):
    stack = rescale(stack,clip=0.3)

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        edge = pool.map(sobel_edge_2d,[tomo for tomo in stack])
        edge = np.where(circular_mask(stack.shape,radius_scale=0.9),np.stack(edge),0)
        edge = pool.starmap(gaussian_blur,[(tomo,sigma) for tomo in edge])
        t = threshold_otsu(np.stack(edge).ravel())
        out = np.zeros_like(stack,dtype=np.uint8)
        out = edge > t
    
    return out


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
