from torchvision import datasets


class CustomData(datasets.ImageFolder):
    """
    CustomData dataset
    """

    def __init__(self, dirpath, transform=None):
        super().__init__(dirpath, transform=transform)
        self.dirpath = dirpath
        self.transform = transform

    def __getitem__(self, index):
        ids_name = self.samples[index][0].split('/')[-1].replace('.png', '')
        imgs, label = super().__getitem__(index)
        return imgs, label, ids_name

