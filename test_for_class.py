from glob import glob
import numpy as np
import pyvox.parser

from sklearn.metrics import accuracy_score, mean_squared_error


from metrics import dice


def __read_vox__(path):
        vox = pyvox.parser.VoxParser(path).parse()
        a = vox.to_dense()
        caja = np.zeros((64, 64, 64))
        caja[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
        return caja

for path_class in sorted(glob('fragmentos_30_100/*')):
    i_class = int(path_class.split('/')[-1])
    class_mse = []
    class_dice = []
    for i_pp in glob(f'{path_class}/*_Dilated.vox'):
        dilated = __read_vox__(i_pp)
        real = __read_vox__(i_pp.replace('Dilated', 'Real'))
        
        error_mse = mean_squared_error(dilated.reshape(1, -1), real.reshape(1, -1))
        class_mse.append(error_mse)
        error_dice = dice(dilated.reshape(64, 64, 64), real.reshape(64, 64, 64))
        class_dice.append(error_dice)
        
        
    print(f'class {i_class}  {np.mean(class_mse)} +/- {np.std(class_mse)}')
    print(f'                 {np.mean(class_dice)} +/- {np.std(class_dice)}')

