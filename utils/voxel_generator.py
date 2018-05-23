import sys
import os
import numpy as np
from memory_profiler import profile


try:
    import trimesh
    from pyvox.models import Vox
    from pyvox.writer import VoxWriter
except:
    print('Intall trimesh and pyvox')

@profile
def getVoxelsFromSTL(path_in, path_out):
    """
    Load stl and scale before transform to voxels.
    Use new np array for cleanup.
    Save .vox at path_out.
    """
    mesh = trimesh.load(path_in)
    mesh.apply_transform(trimesh.transformations.scale_matrix(0.1))
    voxels_save = trimesh.voxel.VoxelMesh(mesh, .7)
    volume = voxels_save.matrix
    print("To export: ", os.path.basename(path_in), volume.shape)
    # del mesh
    # del voxels_save
    res = np.zeros_like(volume, dtype=np.int)
    res[np.nonzero(volume)] = 1
    vox = Vox.from_dense(res)

    name = os.path.basename(path_in)
    VoxWriter(path_out + '%s.vox' %(os.path.splitext(name)[0]),
             vox).write()
    

def STL2VOX_folder(path_in='../output/final_mesh/', 
                   path_out='../output/voxels/'):
    few_files = os.listdir(path_in)
    for file in few_files:
        getVoxelsFromSTL(path_in + '{}'.format(file), path_out)

if __name__ == '__main__':
    STL2VOX_folder()