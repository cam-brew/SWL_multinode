import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import tifffile
from scipy.ndimage import binary_fill_holes
import gc


from skimage.filters import threshold_otsu
from multiprocessing import Pool
from masking_np import circular_mask, masking_blur,masking_edge, close_mask_2, gaussian_blur, isolate_foreground_COM, rescale, keep_largest_component_2d
from segmentation import gaussian_mix_np
from surface_calc import estimate_surface_area
from preproc_img import AAU_process,COM_process
from get_io import get_metadata, clear_dir, read_tomos_dask, write_labels_pool
from pathlib import Path



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



def process_pipeline_AAU(params):
    

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

    core_counts = comm.gather(psutil.cpu_count(logical=False),root=0)

    _,tomo_dir,seg_dir = dirs
    # [stone_id] = stone_id

    log_path = Path(seg_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}_test.txt'
    

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
    
    comm.Barrier()
    shape = comm.bcast(shape,root=0)
    dtype = comm.bcast(dtype,root=0)
    tomo_files = comm.bcast(tomo_files,root=0)
    z_split = comm.bcast(z_split,root=0)
    seg_dir_update = comm.bcast(seg_dir_update,root=0)


    
    z_list = z_split[rank]
    node_files = [tomo_files[i] for i in z_list]

    start = time.time()
    total_start = start

    tomo_stack = read_tomos_dask(node_files)
    with open(log_path,'a+') as f:
        f.write(f'===== Logging for {stone_id} =====\n')
        f.write(f'\nUsing {psutil.cpu_count(logical=False)} cores \nReading {tomo_dir}')
        f.write(f'\nRead {tomo_stack.shape[0]} images: {time.time() - start} seconds\n')
    
    tomo_stack = tomo_stack.compute()
    # circ_slice = np.where(circular_mask(tomo_stack[0].shape,radius_scale=0.9),tomo_stack[0],np.nan)
    valid_slice = circular_mask(tomo_stack[0].shape,radius_scale=0.9).astype(bool)
    print(f'Valid shape: {valid_slice.shape}')
    
    # tomo_stack = np.where(circular_mask(tomo_stack.shape,radius_scale=0.95),tomo_stack,np.nan)

    # tomo_he = rescale(tomo_stack,clip=1) # Histogram equalization
    # del tomo_stack
    # gc.collect()
    
    # blurred = isolate_foreground_AAU_2(tomo_stack,mask_kern_size=25)
    start = time.time()
    blur_mask = np.zeros_like(tomo_stack,dtype=np.uint8)
    if 'Real_05_01' in stone_id:
        blur_mask = masking_blur(tomo_stack,sigma=20,mask=circular_mask(tomo_stack.shape,radius_scale=0.9),keep_largest_comp=True)
    else:
        blur_mask = masking_blur(tomo_stack,sigma=30,mask=circular_mask(tomo_stack.shape,radius_scale=0.9),keep_largest_comp=False)
    print(f'Worker {rank}: completed blur mask {time.time() - start} seconds')

    start = time.time()
    edge_mask = masking_edge(tomo_stack,sigma=3)
    print(f'Worker {rank}: completed edge mask {time.time() - start} seconds')

    mask = np.zeros_like(tomo_stack,dtype=np.uint8)
    mask = np.logical_and(blur_mask,edge_mask)
    
    del blur_mask
    del edge_mask
    gc.collect()

    mask = np.stack(mask).astype(bool)

    with Pool() as pool:
        if 'Real_05_01' in stone_id:
            mask = pool.starmap(close_mask_2,[(img,30,valid_slice,0.9) for img in mask])
        # start = time.time()
        elif 'Real_15_01' in stone_id:
            mask = pool.starmap(close_mask_2,[(img,100,valid_slice,0.5) for img in mask])
        else:
            print(f'Code not yet developed for {stone_id}')
        tomo_clean = pool.starmap(gaussian_blur,[(tomo,2) for tomo in tomo_stack])
        # print(f'Cleaning: {time.time() - start}')


    del tomo_stack
    gc.collect()

    tomo_clean = np.stack(tomo_clean)
    mask = np.stack(mask)

    tomo_clean = rescale(tomo_clean,clip=0.0,mask=mask)

    
    # print(f'Blurring complete')

    # tomo_he,mask = isolate_foreground_AAU_2(tomo_he,blur_kern_size=3,mask_kern_size=50) # Isolate foreground
    
    # ext_sa,ext_faces = estimate_surface_area(mask,1,vox_size=voxel_size) # Retrieve exterior surface area
    # print(f'Worker {rank} gaussian complete')
    
    
    # with open(log_path,'a+') as f:
    #     f.write(f'Histogram adjusted and mask generated in {time.time() - start} seconds\n')
    
    
    # # ====== Section 2: Automated foreground segmentation ======
    
    # print(f'Worker {rank} beginning segmentation')
    # First pass separates solid and fluid
    start = time.time()
    # gmm_stone_labeled,_ = gaussian_mix_np(tomo_clean,mask,n_classes=4) # Label stone (0) and pore (1)
    # gmm_stone_labeled[gmm_stone_labeled == 255] = -1
    gmm_stone_labeled = np.zeros_like(tomo_clean,dtype=np.uint8)
    t = threshold_otsu(tomo_clean[mask].ravel())
    gmm_stone_labeled[mask] = tomo_clean[mask] > t
    print(f'Worker {rank}: Gaussian completed in {time.time() - start} seconds')
    # gmm_stone_labeled = keep_largest_component_3d(gmm_stone_labeled,0) # Eliminate disconnected labels

    

    with Pool(processes=psutil.cpu_count()) as pool:
        gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_labeled)
        gmm_stone_filled = pool.map(keep_largest_component_2d,[tomo for tomo in gmm_stone_filled])


    gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
    # if 'Real_05_01' in stone_id:
    #     print('Isolating largest component')
    #     gmm_stone_filled = keep_largest_component_3d(gmm_stone_filled,id=np.max(gmm_stone_filled),conn=6)
    # else: 
    #     print('Allowing smaller particles')

    gmm_stone_labeled[gmm_stone_filled == 0] = 0
    
    total_sa,total_faces = estimate_surface_area(gmm_stone_labeled,np.max(gmm_stone_labeled),vox_size=voxel_size) # Total surface area
    ext_sa,ext_faces = estimate_surface_area(gmm_stone_filled,np.max(gmm_stone_filled),vox_size=voxel_size)
    # # Interior is total - exterior
    int_sa = total_sa - ext_sa 
    int_faces = total_faces - ext_faces

    # with open(log_path,'a') as f:
    #     f.write(f'\nSolid segmentation: {time.time() - start} seconds')

    
    # print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
   # if air_water_seg == True:
   #     
   #     start = time.time()
   #     pore_stack = gmm_stone_labeled == 1
   #     gmm_pore_labeled,_ = gaussian_mix_np(tomo_clean,pore_stack,n_classes=2)
   #     with open(log_path,'a') as f:
   #         f.write(f'\nFluid segmentation: {time.time() - start} seconds')
   #     
   #     
   #     gmm_integrated = np.zeros_like(tomo_clean,dtype=np.int8)
   #     
   #     gmm_integrated[gmm_integrated == 0] = 0
   #     gmm_integrated[gmm_pore_labeled == 0] = 1 ## Air values set to 1
   #     gmm_integrated[gmm_pore_labeled == 1] = 2 ## Water values set to 2
   #     gmm_integrated[gmm_stone_labeled == 0] = 3 ## Stone values set to 3
   #     if stone_id[:10] == 'Stone_15_01':
   #         gmm_integrated[gmm_pore_labeled == 1] = 3
   #     
   # elif air_water_seg == False:
   #     gmm_integrated = gmm_stone_labeled
   #     gmm_integrated[gmm_integrated == 0] = 0
   #     gmm_integrated[gmm_stone_labeled == 1] = 3 ## Stone values set to 1
   #     print('Stone and fluid labels sorted...')


    # # # ====== Section 3: Generate surface mesh ======
    # # if gen_mesh == True:
    # #     binary = (gmm_integrated == 3).astype(np.uint8)
    # #     start = time.time()
    # #     print('\nGenerating surface mesh')
    # #     mesh,verts,faces = gen_surf(binary,voxel_size,step_size = 1)
    # #     print(f'Generated surface in {time.time() - start} seconds')
        
    # #     start = time.time()
    # #     mesh_sa = mesh_surface_area(verts,faces)
    # #     print(f'Surface area calculated: {mesh_sa} m^2 | {time.time() - start} seconds')
    
    # # ====== Section 3: Stone surface area estimation ======
    # ## Stone surface area calculation
    # print('Estimating surface area (6-conn.)...')
    # start = time.time()
    
    
    # with open(log_path,'a') as f:
    #     f.write(f'\nInterior faces: {int_faces} | Exterior faces: {ext_faces}')
    #     f.write(f'\nInterior SA: {int_sa} mm^2 | Exterior SA: {ext_sa} mm^2')
        

    # # ====== Section 4: Labeled data validation and visualization ======
    # # print('Proceed to developing writing...')
    # # if animate == True:
    # #     animate_stack(node_stack,node_mask,gmm_integrated)
      
    # # # ====== Section 5: Write labeled data to tiff output ======
    
    print(f'Worker {rank} writing labels to {seg_dir_update}')
    start = time.time()
    write_labels_pool(gmm_stone_labeled,seg_dir_update,z_list,prefix=f'/{stone_id}_label_',dtype=gmm_stone_labeled.dtype,cores=None)

    print(f'Worker {rank}: All labels written')
    with open(log_path,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')

    return ext_sa,ext_faces,int_sa,int_faces


