import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import tifffile
from scipy.ndimage import gaussian_filter,binary_fill_holes,binary_erosion,binary_dilation
import gc


from skimage.filters import threshold_otsu
from multiprocessing import Pool
from masking_np import circular_mask, masking_blur_2,masking_post_proc_2,masking_edge, close_mask_2, gaussian_blur, isolate_foreground_COM, rescale, keep_largest_component_3d
from segmentation import gaussian_mix_np,parallel_otsu
from surface_calc import estimate_surface_area
from surface_gen import gen_surf, clean_mesh, simplify_mesh,write_surface
from preproc_img import AAU_process,COM_process
from get_io import get_metadata, clear_dir, read_tomos_dask, write_labels_pool
from pathlib import Path
from mpi4py import MPI

## Interleaved split slices
def split_slices_interleaved(num_slices,node_cores):
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

## Chunked split slices
def split_slices_sequence(num_slices,node_cores):
    total_weight = sum(node_cores)
    weights = [cores/total_weight for cores in node_cores]
    node_slices_counts = [int(round(w * num_slices)) for w in weights]
    diff = num_slices - sum(node_slices_counts)

    for i in range(abs(diff)):
        node_slices_counts[i % len(node_cores)] += np.sign(diff)

    node_slices = []
    start = 0
    for count in node_slices_counts:
        node_slices.append(list(range(start,start + count)))
        start += count
    return node_slices


def process_pipeline_AAU(params):
    

    (
        dirs,
        stone_id,
        voxel_size,
        start_slice,
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

    core_counts = comm.gather(psutil.cpu_count(logical=False),root=0)

    _,tomo_dir,seg_dir = dirs
    dilate_dir = Path(seg_dir).parent.parent / f'{stone_id[:10]}_over' / 'labels/'
    erode_dir = Path(seg_dir).parent.parent / f'{stone_id[:10]}_under' / 'labels/'
    log_path_reg = Path(seg_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}_test.txt'
    log_path_dilate = Path(dilate_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}_test.txt'
    log_path_erode = Path(erode_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}_test.txt'


    if rank == 0:
        seg_dir_update = list(Path(seg_dir).glob(stone_id+f'*/'))
        seg_dir_update = seg_dir_update[0].as_posix()
        print(f'Clearing standard directories in {seg_dir_update}')
        clear_dir(seg_dir_update)

        dilate_dir_update = list(Path(dilate_dir).glob(stone_id+f'*/'))
        dilate_dir_update = dilate_dir_update[0].as_posix()
        print(f'Clearing dilate directories in {dilate_dir_update}')
        clear_dir(dilate_dir_update)

        erode_dir_update = list(Path(erode_dir).glob(stone_id+f'*/'))
        erode_dir_update = erode_dir_update[0].as_posix()
        print(f'Clearing erode directories in {erode_dir_update}')
        clear_dir(erode_dir_update)
        
        tomo_files,shape,dtype = get_metadata(tomo_dir)
        ## remove when testing vull sample
        if start_slice == None:
            start_slice = 0
        if end_slice == None:
            end_slice = -1 
        tomo_files = tomo_files[start_slice:end_slice:skip_interval]
        print(len(tomo_files), core_counts)
        n_slices = len(tomo_files)
        if 'Real_05_01' in stone_id:
            z_split = split_slices_sequence(n_slices,core_counts)
        elif 'Real_15_01' in stone_id:
            z_split = split_slices_interleaved(n_slices,core_counts)
        

    else:
        tomo_files = None
        z_split = None
        shape = None
        dtype = None
        seg_dir_update = None
        dilate_dir_update = None
        erode_dir_update = None
    
    ## Parallel node variable distribution
    comm.Barrier()
    shape = comm.bcast(shape,root=0)
    dtype = comm.bcast(dtype,root=0)
    tomo_files = comm.bcast(tomo_files,root=0)
    z_split = comm.bcast(z_split,root=0)
    seg_dir_update = comm.bcast(seg_dir_update,root=0)
    dilate_dir_update = comm.bcast(dilate_dir_update,root=0)
    erode_dir_update = comm.bcast(erode_dir_update,root=0)


    ## split files for each node according to z_split
    z_list = z_split[rank]
    node_files = [tomo_files[i] for i in z_list]
    start = time.time()
    total_start = start

    tomo_stack = read_tomos_dask(node_files)
    with open(log_path_reg,'a+') as f:
        f.write(f'===== Logging for {stone_id} =====\n')
        f.write(f'\nUsing {psutil.cpu_count(logical=False)} cores \nReading {tomo_dir}')
        f.write(f'\nRead {tomo_stack.shape[0]} images: {time.time() - start} seconds\n')
    
    tomo_stack = tomo_stack.compute()
    valid_slice = circular_mask(tomo_stack[0].shape,radius_scale=0.9).astype(bool)
    print(f'Valid shape: {valid_slice.shape}')
    
    
    ## Foreground masking
    start = time.time()
    mask = np.zeros_like(tomo_stack,dtype=np.uint8)
    if 'Real_05_01' in stone_id:
        mask,tomos_blur,hist,valid,t_otsu = masking_blur_2(tomo_stack,gauss_sigma=20,mask=circular_mask(tomo_stack.shape,radius_scale=0.85),chunk_size=4)
        
        if rank == size - 1:
            mask = masking_post_proc_2(mask,tomos_blur,t_otsu,valid,dilate_iters=20,keep_largest_comp=True,remask=True)
            mask = binary_erosion(mask,structure=np.array([[[0,0,0],[0,0,0],[0,0,0]],
                                                            [[0,0,0],[0,1,0],[0,1,0]],
                                                            [[0,0,0],[0,0,0],[0,0,0]]]),iterations=8)
            
        else:
            mask = masking_post_proc_2(mask,tomos_blur,t_otsu,valid,dilate_iters=8,keep_largest_comp=True)
            mask = binary_dilation(mask,structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                                            [[0,1,0],[1,1,1],[0,1,0]],
                                                            [[0,0,0],[0,1,0],[0,0,0]]]),iterations=8)
            mask = binary_fill_holes(mask)
        mask = np.stack(mask).astype(bool)
        
        
        
    else:
        mask,_,_,_,_ = masking_blur_2(tomo_stack,gauss_sigma=(0,30,30),mask=circular_mask(tomo_stack.shape,radius_scale=0.9),chunk_size=tomo_stack.shape[0])
        mask = mask.astype(bool)
    print(f'Worker {rank}: completed blur mask {time.time() - start} seconds')

    start = time.time()
    
    gc.collect()

    
    if 'Real_05_01' in stone_id:
        with Pool() as pool:
            tomo_clean = pool.starmap(gaussian_blur,[(tomo,2) for tomo in tomo_stack])
    elif 'Real_15_01' in stone_id:
        with Pool() as pool:
            tomo_clean = pool.starmap(gaussian_blur,[(tomo,3) for tomo in tomo_stack])
            mask = pool.map(binary_fill_holes,mask)
            mask = binary_dilation(mask,structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                                            [[0,1,0],[1,1,1],[0,1,0]],
                                                            [[0,0,0],[0,1,0],[0,0,0]]]),iterations=50)
    
    del tomo_stack
    gc.collect()

    tomo_clean = np.stack(tomo_clean)
    mask = np.stack(mask)
    tomo_clean[mask == 0] = 0
    if stone_id[:10] == 'Real_05_01':
        stone_k = 0.25
    elif stone_id[:10] == 'Real_15_01':
        stone_k = 0.25
    else:
        stone_k = 1

    if rank == 0:
        print(f'Tomo shape: {tomo_clean.shape} | Mask shape: {mask.shape} | Valid shape: {valid_slice.shape} | k constant: {stone_k}')

    
    
    # # ====== Section 2: Automated foreground segmentation ======
    
    # First pass separates solid and fluid
    start = time.time()
    gmm_stone_labeled = np.zeros_like(tomo_clean,dtype=np.uint8)
    gmm_stone_over = np.zeros_like(tomo_clean,dtype=np.uint8)
    gmm_stone_under = np.zeros_like(tomo_clean,dtype=np.uint8)

    print(f'Worker {rank}: Starting standard stone segmentation')
    
    
    gmm_stone_labeled,gmm_stone_over,gmm_stone_under = parallel_otsu(tomo_clean,mask,chunk_size=tomo_clean.shape[0],k=stone_k)
    
    ## Standard thresholding
    if 'Real_05_01' in stone_id:
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_labeled)
            gmm_stone_filled = pool.starmap(binary_erosion,[(img,np.array([[0,1,0],[1,1,1],[0,1,0]]),10) for img in gmm_stone_filled])


        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
        gmm_stone_filled = keep_largest_component_3d(gmm_stone_filled,1)
        gmm_stone_filled = binary_dilation(gmm_stone_filled,structure=np.array([[[0,0,0],[0,0,0],[0,0,0]],
                                                                                [[0,1,0],[1,1,1],[0,1,0]],
                                                                                [[0,0,0],[0,0,0],[0,0,0]]]),iterations=5)
    else:
        gmm_stone_filled = binary_fill_holes(gmm_stone_labeled).astype(bool)

    gmm_stone_labeled[gmm_stone_filled == 0] = 0
    gmm_stone_labeled = keep_largest_component_3d(gmm_stone_labeled,1)
    
    print(f'Worker {rank}: begin pores -- standard')
    if air_water_seg == True:
        gmm_pore_labeled = (gmm_stone_filled == 1) & (gmm_stone_labeled == 0)
        t = threshold_otsu(tomo_clean[gmm_pore_labeled].ravel()) ## Standard Otsu on entire pore space
        gmm_integrated = np.zeros_like(gmm_stone_filled,dtype=np.uint8)
        gmm_integrated[gmm_stone_labeled == 1] = 3
        gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean < t)] = 1
        gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean > t)] = 2

        if stone_id[:10] == 'Real_05_01':
            gmm_integrated = keep_largest_component_3d(gmm_integrated,3)
        elif stone_id[:10] == 'Real_15_01':

            pore_mask = np.isin(gmm_integrated,[1,2])
            gmm_pore_labeled[:] = 0
            gmm_pore_labeled[pore_mask] = 1
            gmm_pore_labeled[gmm_stone_filled == 0] = 0

            vals = tomo_clean[gmm_pore_labeled == 1]
            if vals.size > 0:
                t = threshold_otsu(vals.ravel())
                gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean < t)] = 1
                gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean >= t)] = 2
    
    
    ## ======= Surface area and writing ========
    gmm_stone_labeled[gmm_integrated != 3] = 0
    gmm_stone_labeled[gmm_integrated == 3] = 1
    
    if stone_id[:10] == 'Real_05_01':
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_labeled)
        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
    total_sa,total_faces = estimate_surface_area(gmm_stone_labeled,np.max(gmm_stone_labeled),vox_size=voxel_size) # Total surface area
    ext_sa,ext_faces = estimate_surface_area(gmm_stone_filled,np.max(gmm_stone_filled),vox_size=voxel_size)
    # Interior is total - exterior
    int_sa = total_sa - ext_sa 
    int_faces = total_faces - ext_faces

    print(f'Worker {rank} writing labels to {seg_dir_update}')
    start = time.time()
    write_labels_pool(gmm_integrated,seg_dir_update,z_list,prefix=f'/{stone_id}_fill_',dtype=gmm_integrated.dtype,cores=None)

    print(f'Worker {rank}: All standard labels written')
    with open(log_path_reg,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')

    del gmm_stone_labeled
    gc.collect()

    ## =============================== Over estimation labels ================================
    print(f'Worker {rank}: Starting over estimation stone cleaning and pore segmentation')
    if 'Real_05_01' in stone_id:
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_over)
            gmm_stone_filled = pool.starmap(binary_erosion,[(img,np.array([[0,1,0],[1,1,1],[0,1,0]]),10) for img in gmm_stone_filled])

        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
        print(f'Keeping largest component')
        gmm_stone_filled = keep_largest_component_3d(gmm_stone_filled,1)
        gmm_stone_filled = binary_dilation(gmm_stone_filled,structure=np.array([[[0,0,0],[0,0,0],[0,0,0]],
                                                                                [[0,1,0],[1,1,1],[0,1,0]],
                                                                                [[0,0,0],[0,0,0],[0,0,0]]]),iterations=5)
    else:
        gmm_stone_filled = binary_fill_holes(gmm_stone_over).astype(bool)

    gmm_stone_over[gmm_stone_filled == 0] = 0
    gmm_stone_over = keep_largest_component_3d(gmm_stone_over,1)
    
    print(f'Worker {rank}: Begin pores -- over')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        gmm_pore_over = (gmm_stone_filled == 1) & (gmm_stone_over == 0)
        t = threshold_otsu(tomo_clean[gmm_pore_over].ravel()) ## Standard Otsu on entire pore space
        gmm_integrated = np.zeros_like(gmm_stone_filled,dtype=np.uint8)
        gmm_integrated[gmm_stone_over == 1] = 3
        gmm_integrated[(gmm_pore_over == 1) & (tomo_clean < t)] = 1
        gmm_integrated[(gmm_pore_over == 1) & (tomo_clean > t)] = 2

        if stone_id[:10] == 'Real_05_01':
            gmm_integrated = keep_largest_component_3d(gmm_integrated,3)
        elif stone_id[:10] == 'Real_15_01':
            
            pore_mask = np.isin(gmm_integrated,[1,2])
            gmm_pore_over[:] = 0
            gmm_pore_over[pore_mask] = 1
            gmm_pore_over[gmm_stone_filled == 0] = 0

            vals = tomo_clean[gmm_pore_over == 1]
            if vals.size > 0:
                t = threshold_otsu(vals.ravel())
                gmm_integrated[(gmm_pore_over == 1) & (tomo_clean < t)] = 1
                gmm_integrated[(gmm_pore_over == 1) & (tomo_clean >= t)] = 2
    
    gmm_stone_over[gmm_integrated != 3] = 0
    gmm_stone_over[gmm_integrated == 3] = 1
    
    if stone_id[:10] == 'Real_05_01':
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_over)
        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
    total_sa_over,total_faces_over = estimate_surface_area(gmm_stone_over,np.max(gmm_stone_over),vox_size=voxel_size) # Total surface area
    ext_sa_over,ext_faces_over = estimate_surface_area(gmm_stone_filled,np.max(gmm_stone_filled),vox_size=voxel_size)
    # Interior is total - exterior
    int_sa_over = total_sa_over - ext_sa_over
    int_faces_over = total_faces_over - ext_faces_over

    print(f'Worker {rank} writing labels to {dilate_dir_update}')
    start = time.time()
    write_labels_pool(gmm_integrated,dilate_dir_update,z_list,prefix=f'/{stone_id}_over_',dtype=gmm_integrated.dtype,cores=None)

    print(f'Worker {rank}: All dilated labels written')
    with open(log_path_dilate,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')
    
    del gmm_stone_over
    gc.collect()

    ## ======================================== Under estimation labels ================================
    print(f'Worker {rank}: Starting under estimation stone cleaning and pore segmentation')
    if 'Real_05_01' in stone_id:
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_under)
            gmm_stone_filled = pool.starmap(binary_erosion,[(img,np.array([[0,1,0],[1,1,1],[0,1,0]]),10) for img in gmm_stone_filled])

        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
        gmm_stone_filled = keep_largest_component_3d(gmm_stone_filled,1)
        gmm_stone_filled = binary_dilation(gmm_stone_filled,structure=np.array([[[0,0,0],[0,0,0],[0,0,0]],
                                                                                [[0,1,0],[1,1,1],[0,1,0]],
                                                                                [[0,0,0],[0,0,0],[0,0,0]]]),iterations=5)
    else:
        gmm_stone_filled = binary_fill_holes(gmm_stone_under).astype(bool)

    gmm_stone_under[gmm_stone_filled == 0] = 0
    gmm_stone_under = keep_largest_component_3d(gmm_stone_under,1)
    
    print(f'Worker {rank}: Begin pores -- under')
    if air_water_seg == True:
        gmm_pore_under = (gmm_stone_filled == 1) & (gmm_stone_under == 0)
        t = threshold_otsu(tomo_clean[gmm_pore_under].ravel()) ## Standard Otsu on entire pore space
        gmm_integrated = np.zeros_like(gmm_stone_filled,dtype=np.uint8)
        gmm_integrated[gmm_stone_under == 1] = 3
        gmm_integrated[(gmm_pore_under == 1) & (tomo_clean < t)] = 1
        gmm_integrated[(gmm_pore_under == 1) & (tomo_clean > t)] = 2
        if stone_id[:10] == 'Real_05_01':
            gmm_integrated = keep_largest_component_3d(gmm_integrated,3)
        elif stone_id[:10] == 'Real_15_01':
            pore_mask = np.isin(gmm_integrated,[1,2])
            gmm_pore_under[:] = 0
            gmm_pore_under[pore_mask] = 1
            gmm_pore_under[gmm_stone_filled == 0] = 0

            vals = tomo_clean[gmm_pore_under == 1]
            if vals.size > 0:
                t = threshold_otsu(vals.ravel())
                gmm_integrated[(gmm_pore_under == 1) & (tomo_clean < t)] = 1
                gmm_integrated[(gmm_pore_under == 1) & (tomo_clean >= t)] = 2
    
    gmm_stone_under[gmm_integrated != 3] = 0
    gmm_stone_under[gmm_integrated == 3] = 1
    
    if stone_id[:10] == 'Real_05_01':
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_under)
        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
    total_sa_under,total_faces_under = estimate_surface_area(gmm_stone_under,np.max(gmm_stone_under),vox_size=voxel_size) # Total surface area
    ext_sa_under,ext_faces_under = estimate_surface_area(gmm_stone_filled,np.max(gmm_stone_filled),vox_size=voxel_size)
    int_sa_under = total_sa_under - ext_sa_under
    int_faces_under = total_faces_under - ext_faces_under

    print(f'Worker {rank} writing labels to {erode_dir_update}')
    start = time.time()
    write_labels_pool(gmm_integrated,erode_dir_update,z_list,prefix=f'/{stone_id}_under_',dtype=gmm_integrated.dtype,cores=None)

    print(f'Worker {rank}: All eroded labels written')
    with open(log_path_erode,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')
    
    return ext_sa,ext_faces,int_sa,int_faces,ext_sa_over,ext_faces_over,int_sa_over,int_faces_over,ext_sa_under,ext_faces_under,int_sa_under,int_faces_under


