import plotly.graph_objects as go
import numpy as np
import pyvox.parser
import torch
import utils.network_vox as nv
from scipy import ndimage as ndi

available_device = "cuda" if torch.cuda.is_available() else "cpu"


def __read_vox_frag__(path, fragment_idx):
    vox_pottery = __read_vox__(path)
    try:
        assert(fragment_idx in np.unique(vox_pottery))
        vox_frag = vox_pottery.copy()
        vox_frag[vox_pottery != fragment_idx] = 0
        vox_frag[vox_pottery == fragment_idx] = 1
        return vox_frag
    except AssertionError:
        print('fragment_idx not found. Possible fragment_idx {}'.format(
            np.unique(vox_pottery)[1:]))


def __read_vox__(path):
    vox = pyvox.parser.VoxParser(path).parse()
    a = vox.to_dense()
    caja = np.zeros((64, 64, 64))
    caja[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    return caja


def plot(voxel_matrix):
    voxels = np.array(np.where(voxel_matrix)).T
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, symbol='square', color='#ceabb2', line=dict(width=2,
                                                                                                                                     color='DarkSlateGrey',))))
    fig.update_layout()

    fig.show()


def plot_frag(vox_pottery):
    stts = []
    colors = ['#ceabb2', '#d05d86', '#7e1b2f', '#c1375b', '#cdc1c3']
    for i, frag in enumerate(np.unique(vox_pottery)[1:][::-1]):
        vox_frag = vox_pottery.copy()
        vox_frag[vox_pottery != frag] = 0
        voxels = np.array(np.where(vox_frag)).T
        x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
        # ut.plot(vox_frag)
        scatter = go.Scatter3d(x=x, y=y, z=z,
                               mode='markers',
                               name='Fragment {} ({})'.format(i+1, int(frag)),
                               marker=dict(size=5, symbol='square', color=colors[i],
                                           line=dict(width=2, color='DarkSlateGrey',)))
        stts.append(scatter)

    fig = go.Figure(data=stts)
    fig.update_layout()

    fig.show()


def plot_join(vox_1, vox_2):
    stts = []
    colors = ['#ceabb2', '#d05d86', '#7e1b2f', '#c1375b', '#cdc1c3']
    voxels = np.array(np.where(vox_1)).T
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    # ut.plot(vox_frag)
    scatter = go.Scatter3d(x=x, y=y, z=z,
                           mode='markers',
                           name='Fragment 1',
                           marker=dict(size=5, symbol='square', color=colors[0],
                                       line=dict(width=2, color='DarkSlateGrey',)))
    stts.append(scatter)

    voxels = np.array(np.where(vox_2)).T
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    # ut.plot(vox_frag)
    scatter = go.Scatter3d(x=x, y=y, z=z,
                           mode='markers',
                           name='Fragment 2',
                           marker=dict(size=5, symbol='square', color=colors[2],
                                       line=dict(width=2, color='DarkSlateGrey',)))
    stts.append(scatter)

    fig = go.Figure(data=stts)
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
