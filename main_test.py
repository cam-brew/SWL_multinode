import time
import os
import atexit
import signal
import sys
import multiprocessing
import threading

from mpi4py import MPI
from pipeline_AAU import process_pipeline_AAU
from pipeline_COM import process_pipeline_COM
from get_io import get_user_input
from pathlib import Path



def main(comm,param_file):
    
    rank = comm.Get_rank()

    if rank==0:
        
        param_path = param_file
        id_details, task_settings = get_user_input(param_path)

        print(f'Metadata from {param_path}')
        
        
        
        stone_id,voxel_size,end_slice,skip_interval = id_details
        gen_mesh,air_water_seg,animate,num_classes = task_settings

        print(f'Processing Stone {stone_id}')
        
        root_dir = '/data/visitor/me1663/id19/20240227/'
        if stone_id[:10] == "Real_05_01":
            tomo_dir = root_dir + f'PROCESSED_DATA/{stone_id[:10]}/delta_beta_150/Reconstruction_16bit_dff_s32_v2/{stone_id}_16bit_vol/'
        elif stone_id[:10] == "Real_15_01":
            tomo_dir = root_dir + f'PROCESSED_DATA/{stone_id[:10]}/Reconstruction_16bit_dff_s32_v2/{stone_id}_16bit_vol/'
        
        seg_dir = root_dir + f"SEGMENTATION/{stone_id[:10]}/labels/"

        dirs = (root_dir,tomo_dir,seg_dir)
        
        SLURM_CPUS = None
        
        
        task_args = [
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
                SLURM_CPUS,
                comm
            )
        ]
    else:
        task_args=None

    task_args = comm.bcast(task_args,root=0)
    
    ext_sa = None
    ext_faces = None
    int_sa = None
    int_faces = None

    start = time.time()
    if 'Real_05_01' in task_args[0][1]:
        print(f'ID: {task_args[0][1]}')
        ext_sa,ext_faces,int_sa,int_faces = process_pipeline_AAU(*task_args)
    elif 'Real_15_01' in task_args[0][1]:
        print(f'ID {task_args[0][1]}')
        ext_sa,ext_faces,int_sa,int_faces = process_pipeline_AAU(*task_args)
    else:
        print(f'No ID detected in {task_args[0][1]}. Not processing stone')

    print(f'Node {rank} completed in {time.time() - start} seconds')
    
    total_ext_sa = comm.reduce(ext_sa,op=MPI.SUM,root=0)
    total_ext_faces = comm.reduce(ext_faces,op=MPI.SUM,root=0)
    total_int_sa = comm.reduce(int_sa,op=MPI.SUM,root=0)
    total_int_faces = comm.reduce(int_faces,op=MPI.SUM,root=0)


    if rank == 0:
             
        with open(Path(seg_dir).parent / 'data' / f'{stone_id}_test_data.txt', 'w+') as f:
            f.write(f'Surface area information: {stone_id}')
            f.write(f'-------------------------')
            f.write(f'\nTotal measured exterior surface area(mm^2): {total_ext_sa}')
            f.write(f'\nTotal exterior faces detected: {total_ext_faces}')
            f.write(f'\nTotal measured interior surface area (mm^2): {total_int_sa}')
            f.write(f'\nTotal interior faces detected: {total_int_faces}\n')
    
    pass

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    inputs_path = sys.argv[1]
    main(comm,inputs_path)
