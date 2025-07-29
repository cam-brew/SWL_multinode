import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import tifffile
import gc

from masking_np import circular_mask, isolate_foreground_AAU, isolate_foreground_COM, rescale, keep_largest_component_3d
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
    
    # tomo_stack = tomo_stack.compute()
    
    if stone_id[:10] == 'Real_05_01':
        print(f'Beginning processing on {rank}')
        gmm_integrated,surface_data = AAU_process(tomo_stack.compute(),stone_id,voxel_size,air_water_seg,log_path,rank)
        ext_sa,ext_faces,int_sa,int_faces = surface_data
        print(f'Completed processing on {rank}')
    elif stone_id[:10] == 'Real_15_01':
        print(f'Beginning processing on {rank}')
        gmm_integrated,surface_data = COM_process(tomo_stack.compute(),stone_id,voxel_size,air_water_seg,log_path,rank)
        ext_sa,ext_faces,int_sa,int_faces = surface_data
        print(f'Completed processing on {rank}')

    else:
        print(f'Enter valid ID. Received {stone_id[:10]}')

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
    
    
    with open(log_path,'a') as f:
        f.write(f'\nInterior faces: {int_faces} | Exterior faces: {ext_faces}')
        f.write(f'\nInterior SA: {int_sa} mm^2 | Exterior SA: {ext_sa} mm^2')
        

    # ====== Section 4: Labeled data validation and visualization ======
    # print('Proceed to developing writing...')
    # if animate == True:
    #     animate_stack(node_stack,node_mask,gmm_integrated)
      
    # # ====== Section 5: Write labeled data to tiff output ======
    
    print(f'Writing labels to {seg_dir_update}')
    start = time.time()
    write_labels_pool(gmm_integrated,seg_dir_update,z_list,prefix=f'/{stone_id}_labels_',cores = None)
    print(f'Worker {rank}: All labels written')
    with open(log_path,'a+') as f:
        f.write(f'\nTime to write labels: {time.time() - start} seconds\n\n')
        f.write(f'\nTotal time: {time.time() - total_start}')

    return ext_sa,ext_faces,int_sa,int_faces
    
