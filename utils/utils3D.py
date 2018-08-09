import os
import gc
import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import trimesh
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
matplotlib.use('agg')

from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.measure as sk

def save_plot_voxels(voxels, path, iteration):
    print("Plooooot")
    voxels = voxels.__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def getVolumeFromSTL(path, sideLen=32):
    mesh = trimesh.load(path)
    mesh.apply_transform(trimesh.transformations.scale_matrix(0.1))
    voxels_save = trimesh.voxel.VoxelMesh(mesh, 0.5)
    del mesh
    gc.collect()
    volume = voxels_save.matrix
    del voxels_save
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float), 
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1, 
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    gc.collect()
    return volume.astype(np.bool)

def getVFByMarchingCubesSTL(name, voxels, threshold=0.5):
    """Voxel Vertices, faces"""
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    sample_mesh = trimesh.Trimesh(v, f)
    sample_mesh.export('/tmp/{}.stl'.format(name))
    
    return v, f


def plotVoxelVisdom(name, voxels, t, visdom, title):
    v, f = getVFByMarchingCubesSTL(name, voxels, t)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

class VesselsDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        # self.args = args

    def __getitem__(self, index):
        
        volume = getVolumeFromSTL(self.root + self.listdir[index], 64)
        volume = np.asarray(volume, dtype=np.float32)
        
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)

