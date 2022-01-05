
from torchvision import datasets
from PIL import Image
from scipy import stats

class CustomDataGAN(datasets.ImageFolder):
    """
    CustomData dataset
    """

    def __init__(self, dirpath, dirtpath_basetope, transform=None, transform_target=None):
        super().__init__(dirpath, transform=transform_target)
        self.dirpath = dirpath
        self.transform_input = transform
        self.transform_target = transform_target
        self.dirtpath_basetope = dirtpath_basetope
        self.black_image = stats.rv_discrete(name='custm', values=([True, False], [0.5, 0.5]))
        self.flip_image = stats.rv_discrete(name='custm', values=([True, False], [0.5, 0.5]))
        
    def getImage(self, root_tope, root_base):
        
        if(self.flip_image.rvs()):
            tope = Image.open(root_tope)
            base = Image.open(root_base)
        else:
            base = Image.open(root_tope)
            tope = Image.open(root_base)
            
        if(self.black_image.rvs()):
            return base, Image.new('RGB', (128, 128), color = 'white')
        
        return  tope,  base


        

    def __getitem__(self, index):
        ids_name = self.samples[index][0].split('/')[-1].replace('.png', '')
        class_filename = self.samples[index][0].split('/')[-2].replace('.png', '')
        root_base = self.dirtpath_basetope + '{}/{}_tope.png'.format(class_filename, ids_name)
        root_tope = self.dirtpath_basetope + '{}/{}_base.png'.format(class_filename, ids_name)
        img = Image.open(self.samples[index][0])
        try:
            
            tope, base = self.getImage(root_tope, root_base)
            
        except FileNotFoundError:
            tope =  Image.new('RGB', (128, 128), color = 'white')
            base =  Image.new('RGB', (128, 128), color = 'white')
            
        if self.transform_input is not None:
            original_tope = self.transform_target(tope)
            original_base = self.transform_target(base)
            tope = self.transform_input(tope)
            base = self.transform_input(base)
            
            
        
        imgs, label = super().__getitem__(index)
        return imgs, label, ids_name, tope, base, original_tope, original_base
