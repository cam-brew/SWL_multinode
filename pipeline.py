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
from surface_gen import gen_surf, clean_mesh, simplify_mesh,write_surface
from preproc_img import AAU_process,COM_process
from get_io import get_metadata, clear_dir, read_tomos_dask, write_labels_pool
from pathlib import Path


## Interleaved split slices
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

## Chunked split slices
# def split_slices(num_slices,node_cores):
#     total_weight = sum(node_cores)
#     weights = [cores/total_weight for cores in node_cores]
#     node_slices_counts = [int(round(w * num_slices)) for w in weights]
#     diff = num_slices - sum(node_slices_counts)

#     for i in range(abs(diff)):
#         node_slices_counts[i % len(node_cores)] += np.sign(diff)

#     node_slices = []
#     start = 0
#     for count in node_slices_counts:
#         node_slices.append(list(range(start,start + count)))
#         start += count
#     return node_slices


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
    # [stone_id] = stone_id

    log_path = Path(seg_dir).parent / 'logs' / f'{stone_id}' / f'{stone_id}_log_{rank:02d}_test.txt'
    

    if rank == 0:
        seg_dir_update = list(Path(seg_dir).glob(stone_id+'*/'))
        seg_dir_update = seg_dir_update[0].as_posix()
        print(f'Clearing directories in {seg_dir_update}')
        clear_dir(seg_dir_update)
        
        tomo_files,shape,dtype = get_metadata(tomo_dir)
        ## remove when testing vull sample
        if start_slice == None:
            start_slice = 0
        if end_slice == None:
            end_slice = -1
        tomo_files = tomo_files[start_slice:end_slice:skip_interval]
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
    valid_slice = circular_mask(tomo_stack[0].shape,radius_scale=0.9).astype(bool)
    print(f'Valid shape: {valid_slice.shape}')
    
    
    start = time.time()
    blur_mask = np.zeros_like(tomo_stack,dtype=np.uint8)
    if 'Real_05_01' in stone_id:
        blur_mask = masking_blur(tomo_stack,sigma=30,mask=circular_mask(tomo_stack.shape,radius_scale=0.9),keep_largest_comp=True)
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
            mask = pool.starmap(close_mask_2,[(img,40,valid_slice,0.9) for img in mask])
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

    
    
    # # ====== Section 2: Automated foreground segmentation ======
    
    # First pass separates solid and fluid
    start = time.time()
    gmm_stone_labeled = np.zeros_like(tomo_clean,dtype=np.uint8)

    
    

    # thresh_list = np.zeros(tomo_clean.shape[0])
    # for z in range(tomo_clean.shape[0]):
    #     px = tomo_clean[z][mask[z]]
    #     if px.size > 0:
    #         thresh_list[z] = threshold_otsu(px)

    # for z in range(tomo_clean.shape[0]):
    #     slice_mask = mask[z]
    #     if np.any(slice_mask):
    #         gmm_stone_labeled[z][slice_mask] = tomo_clean[z][slice_mask] > thresh_list[z]
    ## Threshold entire dataset here
    t = threshold_otsu(tomo_clean[mask].ravel())
    gmm_stone_labeled[mask] = tomo_clean[mask] > t
    print(f'Worker {rank}: Gaussian completed in {time.time() - start} seconds')

    
    if 'Real_05_01' in stone_id:
        with Pool(processes=psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.map(binary_fill_holes,gmm_stone_labeled)
            # gmm_stone_filled = pool.map(keep_largest_component_2d,[tomo for tomo in gmm_stone_filled])


        gmm_stone_filled = np.stack(gmm_stone_filled).astype(np.uint8)
    else:
        gmm_stone_filled = mask.astype(np.uint8)

    gmm_stone_labeled[gmm_stone_filled == 0] = 0
    
    total_sa,total_faces = estimate_surface_area(gmm_stone_labeled,np.max(gmm_stone_labeled),vox_size=voxel_size) # Total surface area
    ext_sa,ext_faces = estimate_surface_area(gmm_stone_filled,np.max(gmm_stone_filled),vox_size=voxel_size)
    # Interior is total - exterior
    int_sa = total_sa - ext_sa 
    int_faces = total_faces - ext_faces

    
    print('Stone segmentation complete...')
    # Second pass segments fluids (air and water presence)
    if air_water_seg == True:
        with Pool(processes = psutil.cpu_count()) as pool:
            gmm_stone_filled = pool.starmap(close_mask_2,[(tomo,100,None,0.8) for tomo in gmm_stone_filled])

        gmm_stone_filled = np.stack(gmm_stone_filled,axis=0)
        gmm_pore_labeled = (gmm_stone_filled == 1) & (gmm_stone_labeled == 0)
        t = threshold_otsu(tomo_clean[gmm_pore_labeled].ravel())
        gmm_integrated = np.zeros_like(gmm_stone_filled,dtype=np.uint8)
        gmm_integrated[gmm_stone_filled == 1] = 3
        gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean < t)] = 1
        gmm_integrated[(gmm_pore_labeled == 1) & (tomo_clean > t)] = 2


    # # # ====== Section 3: Generate surface mesh ======
    # if gen_mesh == True:
        
    #     start = time.time()
    #     print('\nGenerating surface mesh')
    #     mesh,verts,faces = gen_surf(gmm_integrated,voxel_size,step_size = 2)
    #     mesh = clean_mesh(mesh)
    #     mesh = simplify_mesh(mesh)
    #     print(f'Generated surface in {time.time() - start} seconds')

    #     start = time.time()
    #     surface_png(mesh,path=f'{stone_id}.png')
    #     print(f'Visualized surface in {time.time() - start}')

    #     start = time.time()
    #     write_surface(Path(seg_dir).parent)

        
    # #     start = time.time()
    # #     mesh_sa = mesh_surface_area(verts,faces)
    # #     print(f'Surface area calculated: {mesh_sa} m^2 | {time.time() - start} seconds')
    
        

    # # ====== Section 4: Labeled data validation and visualization ======
    # # print('Proceed to developing writing...')
    # # if animate == True:
    # #     animate_stack(node_stack,node_mask,gmm_integrated)
      
    # # # ====== Section 5: Write labeled data to tiff output ======
    
    print(f'Worker {rank} writing labels to {seg_dir_update}')
    start = time.time()
    write_labels_pool(gmm_integrated,seg_dir_update,z_list,prefix=f'/{stone_id}_fill_',dtype=gmm_integrated.dtype,cores=None)

    print(f'Worker {rank}: All labels written')
    with open(log_path,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')
    # return 0, 0, 0, 0
    return ext_sa,ext_faces,int_sa,int_faces


