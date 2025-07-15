import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import tifffile
import gc

from masking_multidir import circular_mask, test_iso_dask, rescale
from preproc_img import remove_spring, gauss_2d_slices
from segmentation import gaussian_mix_dask
from monitor_performance import animate_stack
from surface_calc import estimate_surface_area, mesh_surface_area
from surface_gen import gen_surf
from get_io import get_metadata, clear_dir, read_tomos_dask, write_labels_dask
from pathlib import Path
from dask.distributed import Client



def split_slices(num_slices,node_cores):
    total_weight = sum(node_cores)
    weights = [cores / total_weight for cores in node_cores]
    node_count = len(node_cores)
    node_slices = [[] for _ in range(node_count)]

    assigned_counts = [0] * node_count
    max_slices = [int(round(w * num_slices)) for w in weights]


    progress = [0.0] * node_count

    for slice_idx in range(num_slices):
        for i in range(node_count):
            if assigned_counts[i] < max_slices[i]:
                progress[i] += weights[i]

        best_node = np.argmax(progress)
        node_slices[best_node].append(slice_idx)
        assigned_counts[best_node] +=1
        progress[best_node] -=1

    return node_slices

def process_pipeline_dist(params):


    (
        dirs,
        stone_id,
        voxel_size,
        end_slice,
        skip_interval,
        gen_mesh,
        air_water_seg,
        animate,
        num_classes, 
        slurm_cpus,
        comm
     ) = params
    
    rank = comm.Get_rank()
    size = comm.Get_size()

    ## ELIMINATE ALL 'TEST_' BEFORE RUNNING COMPLETED CODE

    core_counts = comm.gather(psutil.cpu_count(logical=False),root=0)

    _,tomo_dir,seg_dir = dirs

    log_path = Path(seg_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}.txt'
    

    if rank == 0:
        seg_dir_update = list(Path(seg_dir).glob(stone_id+'*/'))
        seg_dir_update = seg_dir_update[0].as_posix()
        print(f'Clearing directories in {seg_dir_update}')
        clear_dir(seg_dir_update)
        
        tomo_files,shape,dtype = get_metadata(tomo_dir)
        ## remove when testing vull sample
        if end_slice == None:
            end_slice = -1
        tomo_files = tomo_files[:end_slice:skip_interval]
        print(len(tomo_files), core_counts)
        n_slices = len(tomo_files)
        z_split = split_slices(n_slices,core_counts)

    else:
        tomo_files = None
        z_split = None
        shape = None
        dtype = None
        seg_dir_update = None

    shape = comm.bcast(shape,root=0)
    dtype = comm.bcast(dtype,root=0)
    tomo_files = comm.bcast(tomo_files,root=0)
    z_split = comm.bcast(z_split,root=0)
    seg_dir_update = comm.bcast(seg_dir_update,root=0)


    
    z_list = z_split[rank]
    node_files = [tomo_files[i] for i in z_list]

    with open(f'./testing_mn_{rank:02d}.txt', 'w+') as f:
        f.write(f'Node {rank} is utilizing {psutil.cpu_count(logical=False)} cores\n')
        f.write(f'Processing {len(node_files)} slices including {z_list[:10]} ending with {node_files[-1]}')

    start = time.time()
    total_start = start

    with open(f'files.txt','w+') as f:
        f.write(f'Files to read on node {rank}:\n')
        for i in range(len(node_files)):
            f.write(f'{node_files[i].as_posix()}\n')
    
    tomo_stack = read_tomos_dask(node_files)
    with open(log_path,'w') as f:
        f.write(f'===== Logging for {stone_id} =====\n')
        f.write(f'\nUsing {psutil.cpu_count(logical=False)} cores \nReading {tomo_dir}')
        f.write(f'\nRead {tomo_stack.shape[0]} images: {time.time() - start} seconds\n')
    
    tomo_stack = tomo_stack.compute()
    # write_labels_dask(tomo_stack,seg_dir_update,z_list,prefix=f'/{stone_id}_raw_',cores = None)
    ## Remove spring slices
    # spring_idx_min = remove_spring(tomo_stack,global_start=0)

    # print(spring_idx_min)


    tomo_stack = np.where(circular_mask(tomo_stack.shape,radius_scale=0.98),tomo_stack,np.nan)

    print(f'Worker {rank} rescaling histogram')
    tomo_he = rescale(tomo_stack)
    
    del tomo_stack
    gc.collect()

    print(f'Worker {rank} isolating foreground')
    smooth,mask = test_iso_dask(tomo_he,blur_kern_size=5,mask_kern_size=35)
    # mask = mask.compute()

    # tomo_he = gauss_2d_slices(tomo_he,sigma=5)
    # tomo_he = smooth.compute()
    # tomo_he[mask != 1] = np.nan
    print(f'Worker {rank} gaussian complete')
    # write_labels_dask(mask,seg_dir_update,z_list,prefix=f'/{stone_id}_mask_',cores = None)
    # write_labels_dask(tomo_he,seg_dir_update,z_list,prefix=f'/{stone_id}_histeq_',cores = None)
    
    
    with open(log_path,'a+') as f:
        f.write(f'Histogram adjusted and mask generated in {time.time() - start} seconds\n')
    
    
    # ====== Section 2: Automated foreground segmentation ======
    
    ## First pass separates solid and fluid
    start = time.time()
    gmm_stone_labeled,_ = gaussian_mix_dask(smooth,mask,n_classes=2)
    with open(log_path,'a') as f:
        f.write(f'\nSolid segmentation: {time.time() - start} seconds')

    tifffile.imwrite('test_stone_seg.tif',gmm_stone_labeled[gmm_stone_labeled.shape[0] // 2])
    print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        
        start = time.time()
        pore_stack = gmm_stone_labeled == 1
        gmm_pore_labeled,_ = gaussian_mix_dask(tomo_he,pore_stack,n_classes=2)
        with open(log_path,'a') as f:
            f.write(f'\nFluid segmentation: {time.time() - start} seconds')
        tifffile.imwrite('test_pore_seg.tif',gmm_pore_labeled[gmm_pore_labeled.shape[0] // 2])
        
        
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
        
    
    # del gmm_stone_labeled
    # del gmm_pore_labeled
    # gc.collect()

    # # ====== Section 3: Generate surface mesh ======
    # if gen_mesh == True:
    #     binary = (gmm_integrated == 3).astype(np.uint8)
    #     start = time.time()
    #     print('\nGenerating surface mesh')
    #     mesh,verts,faces = gen_surf(binary,voxel_size,step_size = 1)
    #     print(f'Generated surface in {time.time() - start} seconds')
        
    #     start = time.time()
    #     mesh_sa = mesh_surface_area(verts,faces)
    #     print(f'Surface area calculated: {mesh_sa} m^2 | {time.time() - start} seconds')
    
    # ====== Section 3: Stone surface area estimation ======
    ## Stone surface area calculation
    print('Estimating surface area (6-conn.)...')
    start = time.time()
    surface_area_mm2,face_count = estimate_surface_area(gmm_integrated,label=3,vox_size=voxel_size)
    
    with open(log_path,'a') as f:
        f.write(f'\nEstimated surface area (6-conn.): {time.time() - start} seconds')
    
    
    # # ====== Section 4: Labeled data validation and visualization ======
    # print('Proceed to developing writing...')
    # if animate == True:
    #     animate_stack(node_stack,node_mask,gmm_integrated)
    # ## Present Gaussian Mixture selection
    # # plot_gmm_masked_clusters(tomo_stack[tomo_stack.shape[0]//2],mask_stack[mask_stack.shape[0]//2],gmm_stone_labeled[gmm_stone_labeled.shape[0]//2],gmm_stone_model,transparency=0.3)
    # # plot_gmm_masked_clusters(tomo_stack[tomo_stack.shape[0]//2],pore_stack[pore_stack.shape[0]//2],gmm_pore_labeled[gmm_stone_labeled.shape[0]//2],gmm_pore_model,transparency=0.9)
    
    # # ====== Section 5: Write labeled data to tiff output ======
    
    print(f'Writing to path: {seg_dir_update}')
    start = time.time()
    write_labels_dask(gmm_integrated,seg_dir_update,z_list,prefix=f'/{stone_id}_labels_',cores = None)
    with open(log_path,'a') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')


    return surface_area_mm2,face_count
    
