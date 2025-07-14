import os
import dask
import dask.array as da
import shutil
import psutil
import numpy as np
import tifffile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from masking import hist_stretch, isolate_foreground
from dask.delayed import delayed

"""
    I/O functions are defined by:
    - get_user_input(): for allow 
"""

bool_map = {'true': True, 'false': False, 'True': True, 'False': False}

def get_user_input(input_file):
    with open(input_file,'r') as f:
        lines = f.readlines()
        
    split_index = next(i for i,line in enumerate(lines) if '__' in line)
    table_lines = lines[2:split_index]
    data = [line.strip().split() for line in table_lines if line.strip()]
    
    stone_id = np.array([row[0] for row in data])
    vox = np.array([float(row[1]) for row in data])
    end = np.array([int(row[2]) for row in data])
    skip = np.array([int(row[3]) for row in data])
    arrs = (stone_id,vox,end,skip)
    
    settings = lines[split_index + 1:]
    # print(f'Settings: {bool_map.get(settings[0].split()[-1].strip().lower())}')
    render_surf = bool_map.get(settings[0].split()[-1].strip().lower())
    seg_air_water = bool_map.get(settings[1].split()[-1].strip().lower())
    animate = bool_map.get(settings[2].split()[-1].strip().lower())
    
    
    
    n_cluster = 2
    if seg_air_water == True:
        n_cluster = 3
        
    tasks = (render_surf,seg_air_water,animate,n_cluster)
    
    return arrs,tasks



def get_metadata(tomo_dir):
    f_names = sorted([f for f in Path(tomo_dir).iterdir() if f.suffix.lower() in ['.tif','.tiff'] and not '._' in f.name])
    if not f_names:
        raise RuntimeError(f'No TIFF files found in dir: {tomo_dir}')
    sample = tifffile.imread(f_names[0])
    return f_names,sample.shape,sample.dtype




def read_tomos_dask(files,cores = None):
    if cores == None:
        cores = psutil.cpu_count(logical=True)
    print(f'Reading {len(files)} files using {cores} cores')

    sample = tifffile.imread(files[0])
    stack = da.stack([
        da.from_delayed(dask.delayed(tifffile.imread)(f),
        shape = sample.shape,
        dtype=sample.dtype)
        for f in files
    ])

    return stack
    
    

def mask_and_stretch(stack,cores=None,stretch=True):
    if cores ==None:
        cores = psutil.cpu_count(logical=True)
    print(f'Masking dataset...')
    # Mask dataset
    start = time.time()
    mask_stack = stack.map_blocks(isolate_foreground,dtype=bool)
    masks = mask_stack.compute(n_workers=cores,n_threads=1)
    print(f'Masks computed in {time.time() - start:.2f} seconds')

    if stretch == True:
        print(f'Stretching histogram...')
        # Stretch dataset
        start = time.time()
        stretch_stack = da.map_blocks(hist_stretch,stack,dtype=stack.dtype)
        stack = stretch_stack.compute(n_workers=cores,n_thread=1)
        print(f'Histogram stretched in {time.time() - start} seconds')

    return stack,masks


def clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
def _save_tiff(slice_data,path):
    tifffile.imwrite(path,slice_data)
    
        
        
def write_labels_dask(label_stack, output_dir, indices, prefix='label_', dtype=np.int8, cores = None):
    if cores == None:
        cores = psutil.cpu_count(logical=True)

    
    label_stack = label_stack.astype(dtype)
    delayed_tasks = []

    for i in range(len(indices)):
        slice_data = label_stack[i]
        out_path = output_dir + f'{prefix}{indices[i]:05d}.tif'
        task = delayed(_save_tiff)(slice_data,out_path)
        delayed_tasks.append(task)

    dask.compute(*delayed_tasks)
    

def main():
    get_user_input()
    
if __name__ == '__main__':
    main()

