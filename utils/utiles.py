import plotly.graph_objects as go
import numpy as np
import pyvox.parser
import torch
import utils.network_vox as nv
from scipy import ndimage as ndi

available_device = "cuda" if torch.cuda.is_available() else "cpu"


def __read_vox__(path):
    vox = pyvox.parser.VoxParser(path).parse()
    a = vox.to_dense()
    caja = np.zeros((64, 64, 64))
    caja[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    return caja


def plot(voxel_matrix):
    voxels = np.array(np.where(voxel_matrix)).T
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, symbol='square', color='red', line=dict(width=2,
                                                                                                                                 color='DarkSlateGrey',))))
    fig.update_layout()

    fig.show()


def posprocessing(fake, mesh_frag):
    a_p = (mesh_frag > 0.5)
    a_fake = (fake[0] > np.mean(fake[0]))
    #a_fake = (fake[0] > 0.1)
    a_fake = np.array(a_fake, dtype=np.int32).reshape(1, -1)

    diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
    a_fake = ndi.binary_erosion(a_fake.reshape(
        64, 64, 64), diamond, iterations=1)
    _a_p = ndi.binary_erosion(a_p.reshape(64, 64, 64), diamond, iterations=1)

    a_fake = ndi.binary_dilation(
        a_fake.reshape(64, 64, 64), diamond, iterations=1)

    a_p = ndi.binary_dilation(a_p.reshape(64, 64, 64), diamond, iterations=1)
    a_fake = a_fake + _a_p
    #a_fake = (a_fake > 0.5)
    # make a little 3D diamond:
    diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
    dilated = ndi.binary_erosion(
        a_fake.reshape(64, 64, 64), diamond, iterations=1)
    dilated = ndi.binary_dilation(
        a_fake.reshape(64, 64, 64), diamond, iterations=1)

    return a_fake, dilated


def load_generator(path_checkpoint):

    G_encode_decode = nv._G_encode_decode(
        cube_len=64, z_latent_space=128, z_intern_space=136).to(available_device)
    checkpoint = torch.load(path_checkpoint, map_location=available_device)
    G_encode_decode.load_state_dict(checkpoint)
    G_encode_decode = G_encode_decode.eval()

    return G_encode_decode


def generate(model, vox_frag):
    mesh_frag = torch.Tensor(vox_frag).unsqueeze(
        0).float().to(available_device)
    output_g_encode = model.forward_encode(mesh_frag)
    fake = model.forward_decode(output_g_encode)
    fake = fake + (mesh_frag.unsqueeze(1))
    fake = fake.detach().cpu().numpy()
    mesh_frag = mesh_frag.detach().cpu().numpy()
    return fake, mesh_frag