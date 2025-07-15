import numpy as np
from scipy.ndimage import convolve
import skimage

"""Functions for calculating various properties:
    - estimate surface area(): capable of 6-connectivity and 26-connectivity
"""
    
def mesh_surface_area(verts,faces):
    print(f'\nCalculating surface area')
    area = skimage.measure.mesh_surface_area(verts=verts,faces=faces)
    print(f'Surface area calculated to be {area} m^2')
    return area

def estimate_surface_area(segmented,label=1,vox_size=1.0e-3,conn=6):
    """
    Summary: Estimate labeled region within a 3D volume
    Args:
        segmented (ndarray): 3D array of labels
        label (int, optional): Label of the material. (e.g. Stone = 1)
        vox_size (float, optional): Length of voxel edge in m/voxel. (Defaults to 0.001 m)

    Returns:
        surface_area_mm2 (float): Approximate surface area in same units as voxel size (m/voxel)
    """
    # Binary label mask
    binary = (segmented == label).astype(np.int8)
    surface_area_mm2 = 0.0
    
    if conn == 6:
        # Set 6-connectivity kernel for face neighbors
        kernel = np.zeros((3,3,3),dtype=np.int8)
        kernel[1,1,0] = 1
        kernel[1,1,2] = 1
        kernel[1,0,1] = 1
        kernel[1,2,1] = 1
        kernel[0,1,1] = 1
        kernel[2,1,1] = 1
        
        
        neighbor_count = convolve(binary,kernel,mode='constant',cval=0)
        exposed_faces = (6 - neighbor_count)[binary == 1]
        print(f'Detected {exposed_faces.sum()} faces.')
        
        ## Total surface area calculation
        surface_area_mm2 = exposed_faces.sum() * (vox_size**2) * (10**6)
     
    else:
        print(f'Surface area calculation could not be completed. Choose 6-connectivity or render surface.')
        
    return surface_area_mm2, exposed_faces.sum()
    