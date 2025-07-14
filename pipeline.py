import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
import psutil
import gc

from preproc_img import process_tomo_stack,show_comparison
from segmentation import gaussian_mix_dask
from monitor_performance import animate_stack
from surface_calc import estimate_surface_area, mesh_surface_area
from surface_gen import gen_surf
from get_io import get_metadata, clear_dir, mask_and_stretch, read_tomos_dask, write_labels_dask
from pathlib import Path
from dask.distributed import Client


def split_slices(total_slices, core_counts):

    core_counts = np.array(core_counts)
    total_cores = core_counts.sum()
    slots = []
    for i, count in enumerate(core_counts):
        slots.extend([i] * count)
    slots = np.array(slots)
    
    node_assignment = slots[:total_slices]
    
    if len(node_assignment) < total_slices:
        repeats = int(np.ceil(total_slices / len(slots)))
        node_assignment = np.tile(slots,repeats[:total_slices])
        
    node_slices = [[] for _ in range(len(core_counts))]
    for idx, node in enumerate(node_assignment):
        node_slices[node].append(idx)
        
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

    log_path = Path(seg_dir).parent / 'logs' / f'test_{stone_id}' / f'{stone_id}_log_{rank:02d}.txt'
    data_path = Path(seg_dir).parent / 'data' / f'test_{stone_id}_data_{rank:02d}.txt'
    

    if rank == 0:
        seg_dir_update = list(Path(seg_dir).glob('test_'+stone_id+'*/'))
        seg_dir_update = seg_dir_update[0].as_posix() + '/'
        print(f'Clearing directory {seg_dir_update}')
        clear_dir(seg_dir_update)
        
        tomo_files,shape,dtype = get_metadata(tomo_dir)
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


    z_start,z_end = z_split[rank]
    node_files = tomo_files[z_start:z_end]

    print(f'Node {rank} is processing slices {z_start} - {z_end}...')

    start = time.time()
    total_start = start

    tomo_stack = read_tomos_dask(node_files,cores=None)
    with open(log_path,'w') as f:
        f.write(f'===== Logging for {stone_id} =====\n')
        f.write(f'\nUsing {psutil.cpu_count(logical=False)} cores \nReading {tomo_dir}')
        f.write(f'\nRead {tomo_stack.shape[0]} images: {time.time() - start} seconds\n')
    
    
    # Masking
    start = time.time()

    node_stack, node_mask = mask_and_stretch(tomo_stack, cores=None, stretch=False)
    del tomo_stack
    gc.collect()

    with open(log_path,'a+') as f:
        f.write(f'Histogram adjusted and mask generated in {time.time() - start} seconds\n')
    # show_comparison(tomo_stack,mask_stack)
    
    # start = time.time()
    # proc_stack = process_tomo_stack(tomo_stack,num_workers=slurm_cpus)
    # with open(f'{seg_dir}/{stone_id}_log.txt','a') as f:
    #     f.write(f'\nRing removal processing: {time.time() - start} seconds')
    # show_comparison(tomo_stack,proc_stack)
    # tomo_stack = proc_stack
    # del proc_stack
    
    # ====== Section 2: Automated foreground segmentation ======
    
    ## First pass separates solid and fluid
    start = time.time()
    gmm_stone_labeled,gmm_stone_model = gaussian_mix_dask(node_stack,node_mask,n_classes=2)
    with open(log_path,'a') as f:
        f.write(f'\nSolid segmentation: {time.time() - start} seconds')
    print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        print(f'Segmenting fluids...')
        
        start = time.time()
        pore_stack = gmm_stone_labeled == 0
        gmm_pore_labeled,gmm_pore_model = gaussian_mix_dask(node_stack,node_mask,n_classes=num_classes - 1)
        with open(log_path,'a') as f:
            f.write(f'\nFluid segmentation: {time.time() - start} seconds')
        
        ## Integrate into common label set
        print('Sorting labels in ascending order...')
        gmm_integrated = np.zeros_like(node_stack,dtype=np.int8)
        
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_pore_labeled == 0] = 1 ## Air values set to 1
        gmm_integrated[gmm_pore_labeled == 1] = 2 ## Water values set to 2
        gmm_integrated[gmm_stone_labeled == 1] = 3 ## Stone values set to 3
        if stone_id == 'Stone_15_01':
            gmm_integrated[gmm_pore_labeled == 1] = 3
        
    elif air_water_seg == False:
        gmm_integrated = gmm_stone_labeled
        gmm_integrated[gmm_integrated == 0] = 0
        gmm_integrated[gmm_stone_labeled == 1] = 3 ## Stone values set to 1
        print('Stone and fluid labels sorted...')
        
    
    del gmm_stone_labeled
    del gmm_pore_labeled
    gc.collect()

    # # # # ====== Section 3: Generate surface mesh ======
    # # # if gen_mesh == True:
    # # #     binary = (gmm_integrated == 3).astype(np.uint8)
    # # #     start = time.time()
    # # #     print('\nGenerating surface mesh')
    # # #     mesh,verts,faces = gen_surf(binary,voxel_size,step_size = 1)
    # # #     print(f'Generated surface in {time.time() - start} seconds')
        
    # # #     start = time.time()
    # # #     mesh_sa = mesh_surface_area(verts,faces)
    # # #     print(f'Surface area calculated: {mesh_sa} m^2 | {time.time() - start} seconds')
    
    # ====== Section 3: Stone surface area estimation ======
    ## Stone surface area calculation
    print('Estimating surface area (6-conn.)...')
    start = time.time()
    surface_area_mm2 = estimate_surface_area(gmm_integrated,label=3,vox_size=voxel_size)
    
    with open(log_path,'a') as f:
        f.write(f'\nEstimated surface area (6-conn.): {time.time() - start} seconds')
    
    
    # ====== Section 4: Labeled data validation and visualization ======
    print('Proceed to developing writing...')
    if animate == True:
        animate_stack(node_stack,node_mask,gmm_integrated)
    ## Present Gaussian Mixture selection
    # plot_gmm_masked_clusters(tomo_stack[tomo_stack.shape[0]//2],mask_stack[mask_stack.shape[0]//2],gmm_stone_labeled[gmm_stone_labeled.shape[0]//2],gmm_stone_model,transparency=0.3)
    # plot_gmm_masked_clusters(tomo_stack[tomo_stack.shape[0]//2],pore_stack[pore_stack.shape[0]//2],gmm_pore_labeled[gmm_stone_labeled.shape[0]//2],gmm_pore_model,transparency=0.9)
    
    # ====== Section 5: Write labeled data to tiff output ======
    
    # print(f'Writing to path: {seg_dir_update}')
    # start = time.time()
    # write_labels_dask(gmm_integrated,seg_dir_update,z_start,z_end,prefix=f'{stone_id}_labels_',cores = None)
    # with open(log_path,'a') as f:
    #     f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
    #     f.write(f'\nTotal time: {time.time() - total_start}')


    return surface_area_mm2
    
