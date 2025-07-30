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


def kill_remaining_children():
    for child in multiprocessing.active_children():
        print(f'Terminating child process {child.name}')
        child.terminate()
        child.join()
    for t in threading.enumerate():
        if t is not threading.main_thread():
            try:
                print(f'Joining thread {t.name}')
                t.join(timeout=1)
            except Exception:
                pass
    try:
        pgid = os.getpgid(0)
        print(f'[Rank 0] Killing process group {pgid}')
        os.killpg(pgid,signal.SIGTERM)
    except Exception as e:
        print(f'Process group kill failed: {e}')

def list_files_in_dir(path):
    abs_path = os.path.abspath(path)
    files = sorted([f for f in Path(abs_path).iterdir() if f.suffix.lower() in ['.txt'] and not '._' in f.name])
    if not files:
        raise RuntimeError(f'No input files found in {abs_path}')
    else:
        print(f'{len(files)} files found\n')
    return files

def main(comm,param_file):
    
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        
        param_path = param_file
        id_details, task_settings = get_user_input(param_path)

        print(f'Metadata from {param_path}')
        
        
        
        stone_ids,voxel_sizes,end_slices,skip_intervals = id_details
        gen_mesh,air_water_seg,animate,num_classes = task_settings

        print(f'Processing Stone {stone_ids[rank]}')
        
        root_dir = '/data/visitor/me1663/id19/20240227/'
        if stone_ids[rank][:10] == "Real_05_01":
            tomo_dir = root_dir + f'PROCESSED_DATA/{stone_ids[rank][:10]}/delta_beta_150/Reconstruction_16bit_dff_s32_v2/{stone_ids[rank]}_16bit_vol/'
        elif stone_ids[rank][:10] == "Real_15_01":
            tomo_dir = root_dir + f'PROCESSED_DATA/{stone_ids[rank][:10]}/Reconstruction_16bit_dff_s32_v2/{stone_ids[rank]}_16bit_vol/'
        
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
    print(f'{task_args[rank]}')
    ext_sa = None
    ext_faces = None
    int_sa = None
    int_faces = None

    start = time.time()
    if 'Real_05_01' in task_args[rank]:
        print('ID recognized')
        ext_sa,ext_faces,int_sa,int_faces = process_pipeline_AAU(*task_args)
    elif 'Real_15_01' in stone_ids[rank]:
        print('ID recognized')
        ext_sa,ext_faces,int_sa,int_faces = process_pipeline_COM(*task_args)
    else:
        print('No ID detected. Not processing stone')

    print(f'Node {rank} completed in {time.time() - start} seconds')
    
    total_ext_sa = comm.reduce(ext_sa,op=MPI.SUM,root=0)
    total_ext_faces = comm.reduce(ext_faces,op=MPI.SUM,root=0)
    total_int_sa = comm.reduce(int_sa,op=MPI.SUM,root=0)
    total_int_faces = comm.reduce(int_faces,op=MPI.SUM,root=0)


    if rank == 0:
         for i in stone_ids:
             
            with open(Path(seg_dir).parent / 'data' / f'{i}_data.txt', 'w+') as f:
                f.write(f'\nTotal measured exterior surface area(mm^2): {total_ext_sa}')
                f.write(f'\nTotal exterior faces detected: {total_ext_faces}')
                f.write(f'\nTotal measured interior surface area (mm^2): {total_int_sa}')
                f.write(f'\nTotal interior faces detected: {total_int_faces}\n')
    
    pass

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    inputs_path = sys.argv[1]

#    input_files = list_files_in_dir(inputs_path)
    
#    for i,f in enumerate(input_files):
#        print(f'\nFile {i} on Worker {comm.Get_rank()}: {f}\n')
    try:
        main(comm,inputs_path)
        comm.Barrier()
    finally:
        if comm.rank == 0:
            #kill_remaining_children()
            print(f'Completed stone')
    
        sys.exit(0)
