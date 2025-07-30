import gc
import numpy as np
import matplotlib.pyplot as plt
import time

from masking_np import circular_mask,isolate_foreground_AAU,isolate_foreground_COM,keep_largest_component_2d,keep_largest_component_3d,rescale
from segmentation import gaussian_mix_np
from surface_calc import estimate_surface_area


def AAU_process(tomo_stack,stone_id,voxel_size,air_water_seg,log_path,comm):
    rank = comm.Get_rank()
    tomo_stack = np.where(circular_mask(tomo_stack.shape,radius_scale=0.95),tomo_stack,np.nan)

    tomo_he = rescale(tomo_stack) # Histogram equalization
    del tomo_stack
    gc.collect()
    
    tomo_he,mask = isolate_foreground_AAU(tomo_he,blur_kern_size=3,mask_kern_size=50) # Isolate foreground
    ext_sa,ext_faces = estimate_surface_area(mask,1,vox_size=voxel_size) # Retrieve exterior surface area
    print(f'Worker {rank} gaussian complete')
    
    
    with open(log_path,'a+') as f:
        f.write(f'Histogram adjusted and mask generated in {time.time() - start} seconds\n')
    
    
    # ====== Section 2: Automated foreground segmentation ======
    
    print(f'Worker {rank} beginning segmentation')
    ## First pass separates solid and fluid
    start = time.time()
    gmm_stone_labeled,_ = gaussian_mix_np(tomo_he,mask,n_classes=2) # Label stone (0) and pore (1)
    gmm_stone_labeled = keep_largest_component_3d(gmm_stone_labeled,0) # Eliminate disconnected labels

    total_sa,total_faces = estimate_surface_area(gmm_stone_labeled,0,vox_size=voxel_size) # Total surface area
    # Interior is total - exterior
    int_sa = total_sa - ext_sa 
    int_faces = total_faces - ext_faces
    surface_data = ext_sa,ext_faces,int_sa,int_faces

    with open(log_path,'a') as f:
        f.write(f'\nSolid segmentation: {time.time() - start} seconds')

    
    print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        
        start = time.time()
        pore_stack = gmm_stone_labeled == 1
        gmm_pore_labeled,_ = gaussian_mix_np(tomo_he,pore_stack,n_classes=2)
        with open(log_path,'a') as f:
            f.write(f'\nFluid segmentation: {time.time() - start} seconds')
        
        
        gmm_integrated = np.zeros_like(tomo_he,dtype=np.int8)
        
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_pore_labeled == 0] = 1 ## Air values set to 1
        gmm_integrated[gmm_pore_labeled == 1] = 2 ## Water values set to 2
        gmm_integrated[gmm_stone_labeled == 0] = 3 ## Stone values set to 3
        if stone_id == 'Stone_15_01':
            gmm_integrated[gmm_pore_labeled == 1] = 3
        
    elif air_water_seg == False:
        gmm_integrated = gmm_stone_labeled
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_stone_labeled == 1] = 3 ## Stone values set to 1
        print('Stone and fluid labels sorted...')

    return gmm_integrated,surface_data

def COM_process(tomo_stack,stone_id,voxel_size,air_water_seg,log_path,comm):
    rank = comm.Get_rank()
    tomo_stack = np.where(circular_mask(tomo_stack.shape,radius_scale=0.95),tomo_stack,np.nan)

    tomo_he = rescale(tomo_stack) # Histogram equalization
    del tomo_stack
    gc.collect()
    
    

    tomo_he,mask = isolate_foreground_COM(tomo_he,blur_kern_size=None,mask_kern_size=15) # Isolate foreground
    comm.Barrier()
    ext_sa,ext_faces = estimate_surface_area(mask,1,vox_size=voxel_size) # Retrieve exterior surface area
    print(f'Worker {rank} gaussian complete')
    
    
    # with open(log_path,'a+') as f:
    #     f.write(f'Histogram adjusted and mask generated in {time.time() - start} seconds\n')
    
    
    # ====== Section 2: Automated foreground segmentation ======
    
    ## First pass separates solid and fluid
    print(f'Worker {rank} beginning segmentation')
    start = time.time()
    gmm_stone_labeled,_ = gaussian_mix_np(tomo_he,mask,n_classes=2) # Label stone (0) and pore (1)
    
    print(f'Beginning surface area estimation')
    total_sa,total_faces = estimate_surface_area(gmm_stone_labeled,0,vox_size=voxel_size) # Total surface area
    # Interior is total - exterior
    int_sa = total_sa - ext_sa 
    int_faces = total_faces - ext_faces
    surface_data = ext_sa,ext_faces,int_sa,int_faces

    # with open(log_path,'a') as f:
    #     f.write(f'\nSolid segmentation: {time.time() - start} seconds')

    
    print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        
        start = time.time()
        pore_stack = gmm_stone_labeled == 1
        gmm_pore_labeled,_ = gaussian_mix_np(tomo_he,pore_stack,n_classes=2)
        # with open(log_path,'a') as f:
        #     f.write(f'\nFluid segmentation: {time.time() - start} seconds')
        
        
        gmm_integrated = np.zeros_like(tomo_he,dtype=np.int8)
        
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_pore_labeled == 0] = 1 ## Air values set to 1
        gmm_integrated[gmm_pore_labeled == 1] = 2 ## Water values set to 2
        gmm_integrated[gmm_stone_labeled == 0] = 3 ## Stone values set to 3
        if stone_id == 'Stone_15_01':
            gmm_integrated[gmm_pore_labeled == 1] = 3
        
    elif air_water_seg == False:
        gmm_integrated = gmm_stone_labeled
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_stone_labeled == 1] = 3 ## Stone values set to 1
        print('Stone and fluid labels sorted...')

    return gmm_integrated,surface_data
def show_comparison(original, cleaned, slice_idx=None, vmin=None, vmax=None):
    """Show before/after comparison of a selected slice."""
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Slice")
    ax[0].axis('off')
    ax[1].imshow(cleaned[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title("Cleaned Slice")
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    
