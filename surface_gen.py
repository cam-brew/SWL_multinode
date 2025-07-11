import trimesh
from skimage.measure import marching_cubes

def write_surface(root_dir,stone_id,mesh):
    output_path = root_dir + f'Visualizations/{stone_id}_surface.stl'
    mesh.export(output_path,file_type='stl')

def gen_surf(binary,vox_size,step_size=1):
    verts,faces,_,_ = marching_cubes(binary,spacing=(vox_size,vox_size,vox_size),gradient_direction='ascent',method='lewiner',allow_degenerate=False,step_size=step_size)
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    return mesh,verts,faces

def clean_mesh(mesh):
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.merge_vertices()
    mesh.fix_normals()
    
    return mesh

def simplify_mesh(mesh):
    mesh = mesh.simplify_quadratic_decimation(target_face_count=2_000_000)
    return mesh