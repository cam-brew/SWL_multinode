import time
import os
import psutil
from mpi4py import MPI
from pipeline import process_pipeline_dist
from get_io import get_user_input
from pathlib import Path

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        
        param_path = '/home/esrf/cameron15a/Desktop/python/inputs/Real_05_01/seg_param_0007.txt'
        id_details, task_settings = get_user_input(param_path)

        print(f'Metadata from {param_path}')
        
        
        
        stone_ids,voxel_sizes,end_slices,skip_intervals = id_details
        gen_mesh,air_water_seg,animate,num_classes = task_settings

        print(f'Processing Stone {stone_ids[rank]}')
        
        root_dir = '/data/visitor/me1663/id19/20240227/'
        tomo_dir = root_dir + f'PROCESSED_DATA/{stone_ids[rank][:10]}/delta_beta_150/Reconstruction_16bit_dff_s32_v2/{stone_ids[rank]}_16bit_vol/'
        seg_dir = root_dir + f"SEGMENTATION/{stone_ids[rank][:10]}_multinode/labels/"

        dirs = (root_dir,tomo_dir,seg_dir)
        
        SLURM_CPUS = None
        
        
        task_args = [
            (
                dirs,
                stone_ids[i],
                voxel_sizes[i],
                end_slices[i],
                skip_intervals[i],
                gen_mesh,
                air_water_seg,
                animate,
                num_classes,
                SLURM_CPUS,
                comm
            )
            for i in range(len(stone_ids))
        ]
    else:
        task_args=None

    task_args = comm.bcast(task_args,root=0)
    # with open(f'./testing_mn_{rank}.txt','w') as f:
    #         f.write(f'Reading from {task_args[0][1]} \nWriting to {task_args[0][2]}\nUsing {psutil.cpu_count(logical=True)} cores on node {rank}\n')

    print(f'Worker {rank} ready')

    start = time.time()

    surface_area_mm2,face_count = process_pipeline_dist(*task_args)

    print(f'Node {rank} completed in {time.time() - start} seconds')
    
    total_SA_mm2 = comm.reduce(surface_area_mm2,op=MPI.SUM,root=0)
    total_faces = comm.reduce(face_count,op=MPI.SUM,root=0)
    if rank == 0:
         for i in stone_ids:
             
            with open(Path(seg_dir).parent / 'data' / f'{i}_data.txt', 'a+') as f:
                f.write(f'Total measured surface area(mm^2): {total_SA_mm2}')
                f.write(f'Total faces detected: {total_faces}')
            print(f'Total surface area (mm^2): {total_SA_mm2}')
            print(f'Total faces detected: {total_faces}')

if __name__ == '__main__':
    
    main()
