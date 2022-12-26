import numpy as np
import torch
from torch.utils import data

Z_LATENT_SPACE = 128
Z_INTERN_SPACE = 136#160#136 #256

#G_LR = 0.0002
#D_LR = 0.001
G_LR = 0.00002 
D_LR = 0.000002
EPOCHS = 5000
BSIZE = 32
CUBE_LEN = 64
BETAS = (0.9, 0.999)

from metrics import dice

from utils.FragmentDatasetCSV import FragmentDatasetCSV
import utils.network_vox as nv
from pyvox.models import Vox
from pyvox.writer import VoxWriter
from sklearn.metrics import accuracy_score, mean_squared_error
from skimage.metrics import hausdorff_distance, structural_similarity



available_device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import sh
from scipy import ndimage as ndi


G_encode_decode = nv._G_encode_decode(z_latent_space=Z_LATENT_SPACE, z_intern_space=Z_INTERN_SPACE).to(available_device)
#dt = FragmentDataset('./data', 'test', )
name_csv_fragmentos = 'fragmentos_15_20'
#name_csv_fragmentos = 'fragmentos_20_30'
#name_csv_fragmentos = 'fragmentos_30_100'
name_csv = f'data/{name_csv_fragmentos}.csv'
dt = FragmentDatasetCSV('./data', 'test', name_csv)


#PATH = 'pesos_guardados/v2_G_encode_decode_partial_360.pkl'

PATH = '../../vox_gan/weight/v2_G_encode_decode_partial_15.pkl'
#PATH = '../../vox_gan/weight/v2_G_encode_decode_partial_15.pkl'

print(PATH, name_csv)
checkpoint = torch.load(PATH)
G_encode_decode.load_state_dict(checkpoint)
G_encode_decode = G_encode_decode.eval()

data_loader = data.DataLoader(dt, batch_size=BSIZE, shuffle=False, drop_last=True)
lista_dice = []
lista_acc = []
lista_hausdorff= []
#lista_structural_similarity=[]
name_csv_fragmentos = name_csv_fragmentos + '_vis'

for i,  (mesh_frag, mesh_complete, label, img_path) in enumerate(data_loader):
    
        mesh_frag = mesh_frag.float().to(available_device)
        mesh_complete = mesh_complete.float().to(available_device)
        output_g_encode = G_encode_decode.forward_encode(mesh_frag)
        fake = G_encode_decode.forward_decode(output_g_encode) 
        fake_shape = fake.detach().cpu().numpy().shape
        fake_numpy = fake.detach().cpu().numpy()
        mesh_frag_numpy = mesh_frag.detach().cpu().numpy().reshape(fake_shape)
        
        mesh_complete_numpy = mesh_complete.detach().cpu().numpy().reshape(fake_shape)

        
        fake_iterator = zip(fake_numpy, mesh_complete_numpy, mesh_frag_numpy, label, img_path)
       

        for i, (f, m, p, l, path) in enumerate(fake_iterator):
            a_fake = (f[0] > np.mean(f[0]))
            a_real = (m[0] > 0.5)
            a_p = (p[0] > 0.5)
            a_real = np.array(a_real, dtype=np.int32).reshape(1, -1)
            a_fake = np.array(a_fake, dtype=np.int32).reshape(1, -1)
            
            diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
            a_fake = ndi.binary_erosion(a_fake.reshape(64, 64, 64), diamond, iterations=1)
            _a_p = ndi.binary_erosion(a_p.reshape(64, 64, 64), diamond, iterations=1)
            #a_fake = a_fake + _a_p
            a_fake = ndi.binary_dilation(a_fake.reshape(64, 64, 64), diamond, iterations=1)
            
            a_p = ndi.binary_dilation(a_p.reshape(64, 64, 64), diamond, iterations=1)
            a_fake = a_fake + _a_p
            a_fake = (a_fake > 0.5)
            
        
            # make a little 3D diamond:
            diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
            dilated = ndi.binary_erosion(a_fake.reshape(64, 64, 64), diamond, iterations=1)
            
            dilated = ndi.binary_dilation(a_fake.reshape(64, 64, 64), diamond, iterations=1)
                        
            #print(np.unique(dilated))
            
            dilated = ndi.binary_erosion(dilated.reshape(64, 64, 64), diamond, iterations=1)
        
            dilated_color = np.array(dilated, dtype=np.int32)
            dilated_color[a_p] = dilated_color[a_p] * 2
            dilated_color = dilated_color * 2
            
            a_p = np.array(a_p, dtype=np.int32) * 4
            
            
            a_real_vox = Vox.from_dense(a_real.reshape(64, 64, 64))
            a_p_vox = Vox.from_dense(a_p.reshape(64, 64, 64))
            dilated_vox = Vox.from_dense(dilated.reshape(64, 64, 64))
            dilated_color_vox = Vox.from_dense(dilated_color.reshape(64, 64, 64))
            
            name_file = path.split('/')[-1].replace('.vox','')
            
            sh.mkdir('-p', f'{name_csv_fragmentos}/{l}/')
            
        
            VoxWriter(f'{name_csv_fragmentos}/{l}/model_{name_file}_Dilated.vox', dilated_vox).write()
            VoxWriter(f'{name_csv_fragmentos}/{l}/model_{name_file}_Real.vox', a_real_vox).write()
            VoxWriter(f'{name_csv_fragmentos}/{l}/model_{name_file}_Fragment.vox', a_p_vox).write()
            VoxWriter(f'{name_csv_fragmentos}/{l}/model_{name_file}_D_Color.vox', dilated_color_vox).write()
                  
            lista_acc.append(mean_squared_error(a_real.reshape(1, -1), dilated.reshape(1, -1)))
            lista_dice.append(dice(a_real.reshape(64, 64, 64), dilated.reshape(64, 64, 64)))
            #break
            
                
print(f'acc, {np.mean(lista_acc)}, +/- {np.std(lista_acc)}')
print(f'dice, {np.mean(lista_dice)}, +/- {np.std(lista_dice)}')
