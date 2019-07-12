import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import bisect
import warnings
import pandas as pd
import shutil
from torch._utils import _accumulate
from torch import randperm, utils
from torchvision import datasets, transforms
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import manifold


class Subset(utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def create_df_from_files(path='data/perfiles_CATA/clases/'):
    l = list()
    files = sorted(os.listdir(path), key=lambda i: int(os.path.splitext(i)[0]))
    for class_, filename in enumerate(files, 1):
        print(class_)
        with open(os.path.join(path, filename)) as f:
            lines = f.readlines()
            #print(lines)
            for id_ in lines:
                l.append((id_.rstrip(), class_))
    df_classes = pd.DataFrame(l, columns=['id', 'class'])

    return df_classes

def create_folder_pytorch_format(df, destination, path):
    for row in df.iterrows():
        directory = os.path.join(destination, str(row[1][1]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        name = row[1][0] + '.png'
        for root, dirs, files in os.walk(path):
            if name in files:
                print(os.path.join(root, name))
                shutil.copy(os.path.join(root, name), destination + str(row[1][1]))

def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def proj(X, ax, ax2d):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        tr = ax.transData.transform((x2, y2))
        return ax2d.transData.inverted().transform(tr)


def plot_tsne_3D(X_tsne, merged, azim=120, distance=70000):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax2d = fig.add_subplot(111,frame_on=False) 
    ax2d.axis("off")
    ax.view_init(elev=30., azim=azim)
    for i in range(X_tsne.shape[0]):
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], c=plt.cm.magma(merged.iloc[i][1] / 11.), s=100)
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1., 1.]])
        for i in range(merged.shape[0]):           
            dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)

            if np.min(dist) < distance:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X_tsne[i]]]
            image =  Image.open('data/perfiles_CATA/png_full/' + merged.iloc[i][0] + '.png')
            inverted_image = PIL.ImageOps.invert(image)
            inverted_image.thumbnail((40, 40), Image.ANTIALIAS)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image),
                proj(X_tsne[i], ax, ax2d))
            ax2d.add_artist(imagebox)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('t-SNE over the 11 classes of vessels')
    plt.savefig("/tmp/movie%d.png" % azim)

# Quizas deberia eliminar 3d o limpiar
def plot_embedding(X, merged, title, classes):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(merged.iloc[i][1]),
                 color=plt.cm.Set1(int(merged.iloc[i][1]) / float(classes)),
                 fontdict={'weight': 'bold', 'size': 9})
            
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(merged.shape[0]):           
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 6e-3:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X[i]]]
            #image =  Image.open('data/perfiles_CATA/png_full/' + merged.iloc[i][0] + '.png')
            image =  Image.open(merged.iloc[i][0])
            inverted_image = PIL.ImageOps.invert(image)
            inverted_image.thumbnail((30, 30), Image.ANTIALIAS)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

if __name__ == "__main__":
    pass