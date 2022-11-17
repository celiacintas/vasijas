from torchvision.io import read_image
import glob
from torch.utils.data import Dataset
import trimesh
import numpy as np
import pyvox.parser

class FragmentDataset(Dataset):
    def __init__(self, vox_path, vox_type, dim_size=64, transform=None):
        self.vox_type = vox_type
        self.vox_path = vox_path
        self.transform = transform
        self.dim_size = dim_size
        self.vox_files = sorted(glob.glob('{}/{}/*/*.vox'.format(self.vox_path, self.vox_type)))

    def __len__(self):
        return len(self.vox_files)

    def __read_vox__(self, path):
        vox = pyvox.parser.VoxParser(path).parse()
        a = vox.to_dense()
        caja = np.zeros((64, 64, 64))
        caja[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
        return caja
    
    def __select_fragment__(self, v):
        frag_id = np.unique(v)[1:]
        #select_frag = np.random.choice(frag_id, np.random.choice(np.arange(1, len(frag_id)), 1)[0], replace=False)
        select_frag = np.random.choice(frag_id, 1, replace=False)
        print(select_frag)
        for f in frag_id:
            if not(f in select_frag):
                v[v==f] = 0
            else:
                v[v==f] = 1
        return v, select_frag
    
    def __non_select_fragment__(self, v, select_frag):
        frag_id = np.unique(v)[1:]
        for f in frag_id:
            if not(f in select_frag):
                v[v==f] = 1
            else:
                v[v==f] = 0
        return v
    
    def __select_fragment_specific__(self, v, select_frag):
        frag_id = np.unique(v)[1:]
        for f in frag_id:
            if not(f in select_frag):
                v[v==f] = 0
            else:
                v[v==f] = 1
        return v, select_frag


    def __getitem__(self, idx):
        img_path = self.vox_files[idx]
        vox = self.__read_vox__(img_path)
        label = img_path.replace(self.vox_path, '').split('/')[2]
        frag, select_frag= self.__select_fragment__(vox.copy())
        
        if self.transform:
            vox = self.transform(vox)
            frag = self.transform(frag)

        return frag, vox, #select_frag, int(label)-1#, img_path
    
    def __getitem_specific_frag__(self, idx, select_frag):
        img_path = self.vox_files[idx]
        vox = self.__read_vox__(img_path)
        label = img_path.replace(self.vox_path, '').split('/')[2]
        frag, select_frag= self.__select_fragment_specific__(vox.copy(), select_frag)
        
        if self.transform:
            vox = self.transform(vox)
            frag = self.transform(frag)

        return frag, vox, #select_frag, int(label)-1, img_path
    
    def __getfractures__(self, idx):
        img_path = self.vox_files[idx]
        vox = self.__read_vox__(img_path)
        return np.unique(vox) #select_frag, int(label)-1, img_path