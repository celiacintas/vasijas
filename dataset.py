# %%
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pyvox.parser
from glob import glob


class FragmentDatasetCSV(Dataset):
   
    def __init__(self, vox_path, vox_type, csv_file, dim_size=64, transform=None):
        self.vox_type = vox_type
        self.vox_path = vox_path
        self.transform = transform
        self.dim_size = dim_size
        self.vox_files = pd.read_csv(csv_file)
    
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
        select_frag = np.random.choice(frag_id, np.random.choice(np.arange(1, len(frag_id)), 1)[0], replace=False)
        for f in frag_id:
            if not(f in select_frag):
                v[v==f] = 0
            else:
                v[v==f] = 1
        return v, select_frag
    
    def __select_specifics_fragment__(self, v, select_frag):
        frag_id = np.unique(v)[1:]
        for f in frag_id:
            if not(f == select_frag):
                v[v==f] = 0
            else:
                v[v==f] = 1
        return v, select_frag
    
    def __non_select_fragment__(self, v, select_frag):
        frag_id = np.unique(v)[1:]
        for f in frag_id:
            if not(f == select_frag):
                v[v==f] = 1
            else:
                v[v==f] = 0
        return v


    def __getitem__(self, idx):

        row = self.vox_files.iloc[idx]
        file = row['Fichero']
        img_path = sorted(glob('{}/{}/*/{}'.format(self.vox_path, self.vox_type, file)))[0]
        
        vox = self.__read_vox__(img_path)
        label = img_path.replace(self.vox_path, '').split('/')[2]
        frag, select_frag= self.__select_specifics_fragment__(vox.copy(), int(row[' Fragment id']))
        
        if self.transform:
            vox = self.transform(vox)
            frag = self.transform(frag)

        #return frag, vox, select_frag, int(label)-1, img_path
        return frag, vox, select_frag#, int(label)-1, img_path
    
# %%
dt = FragmentDatasetCSV('./data', 'test', 'data/fragmentos_30_100.csv')

# %%
dt.__getitem__(0)
# %%
