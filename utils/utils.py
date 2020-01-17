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
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib

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
        #print(class_)
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


def plot_tsne_3D(X_tsne, merged, azim=120, distance=7000):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax2d = fig.add_subplot(111,frame_on=False) 
    ax2d.axis("off")
    ax.view_init(elev=30., azim=azim)
    for i in range(X_tsne.shape[0]):
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color=plt.cm.magma(merged.iloc[i][1] / 11.), s=100)
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
            props = dict(boxstyle='round', facecolor='red')
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image),
                proj(X_tsne[i], ax, ax2d), bboxprops=props)
            ax2d.add_artist(imagebox)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('t-SNE over the 11 classes of vessels')
    plt.savefig("/tmp/movie%d.png" % azim)



# Quizas deberia eliminar 3d o limpiar
def plot_embedding(X, merged, title = None, classes=11., showimage=True):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()
    ax = plt.subplot(111)
    ax.set_facecolor('xkcd:white')
    """
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(merged.iloc[i][1]),
                 color=plt.cm.Set1(int(merged.iloc[i][1]) / float(classes)),
                 fontdict={'weight': 'bold', 'size': 9})
    """
    for i in range(X.shape[0]):
        plt.plot([X[i, 0]], [X[i, 1]], 'o', c="black", markersize=8)
        plt.plot([X[i, 0]], [X[i, 1]], 'o',c=plt.cm.Set3(int(merged.iloc[i][1])), markersize=6)
    
  
    
    if showimage and hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(merged.shape[0]):           
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 6e-3:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X[i]]]
            image =  Image.open(merged.iloc[i][0])
            inverted_image = image #PIL.ImageOps.invert(image)
            inverted_image.thumbnail((40, 40), Image.ANTIALIAS)
            props = dict(facecolor='white', alpha=1, lw=1)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image, cmap=plt.cm.gray),
                X[i]+0.025, bboxprops=props)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    #cbar = plt.colorbar()
    if title is not None:
        plt.title(title)

"""
# Quizas deberia eliminar 3d o limpiar
def plot_embedding(X, merged, title = None, classes=10.):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    ax.set_facecolor('xkcd:white')
    
    for i in range(X.shape[0]):
        plt.plot([X[i, 0]], [X[i, 1]], 'o', c="black", markersize=8)
        plt.plot([X[i, 0]], [X[i, 1]], 'o',c=plt.cm.Greens(int(merged.iloc[i][1]) / float(classes)), markersize=6)
        #plt.plot([X[i, 0]], [X[i, 1]], 'o',c=plt.cm.Greens(int(merged.iloc[i][1]) / float(classes)), markersize=4)
 
    
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(merged.shape[0]):           
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 6e-3:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X[i]]]
            image =  Image.open(merged.iloc[i][0])
            inverted_image = image #PIL.ImageOps.invert(image)
            inverted_image.thumbnail((40, 40), Image.ANTIALIAS)
            props = dict(facecolor='white', alpha=1, lw=1)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image, cmap=plt.cm.gray),
                X[i]+0.015, bboxprops=props)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


"""

def iterations_test_pixel(test_loader, available_device = "cpu"):
    y_real = list()
    y_pred = list()
    for ii, data_ in enumerate(test_loader):
        input_, label = data_
        val_input = Variable(input_).to(available_device)
        val_label = Variable(label.type(torch.LongTensor)).to(available_device)
        score = val_input
        y_pred_batch = score.detach().cpu().squeeze().numpy()
        y_real_batch = val_label.cpu().data.squeeze().numpy()
        y_real.append(y_real_batch.tolist())
        y_pred.append(y_pred_batch.tolist())

    y_real = [item for batch in y_real for item in batch]
    y_pred = [item for batch in y_pred for item in batch]
    
    return y_real, y_pred

def plot_confusion_matrix(y_true, y_pred, classes,
                          filename,normalize=False,
                          title="",cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = ''
        else:
            title = ''

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    #im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im = ax.imshow(cm, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='',
           xlabel='')
    
    ax.grid(False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i in range(cm.shape[0]):
        #for j in range(cm.shape[1]):
            #ax.text(j, i, format(cm[i, j], fmt),
            #        ha="center", va="center",
            #        color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    return ax

def iterations_test(C, test_loader, available_device = "cpu"):
    y_real = list()
    y_pred = list()

    for ii, data_ in enumerate(test_loader):
        input_, label = data_
        val_input = Variable(input_).to(available_device)
        val_label = Variable(label.type(torch.LongTensor)).to(available_device)
        score = C(val_input)
        _, y_pred_batch = torch.max(score, 1)
        y_pred_batch = y_pred_batch.cpu().squeeze().numpy()
        y_real_batch = val_label.cpu().data.squeeze().numpy()
        y_real.append(y_real_batch.tolist())
        y_pred.append(y_pred_batch.tolist())

    y_real = [item for batch in y_real for item in batch]
    y_pred = [item for batch in y_pred for item in batch]
    
    return y_real, y_pred

def iterations_test_partial(C, test_loader, available_device = "cpu"):
    y_real = list()
    y_pred = list()
    for ii, data_ in enumerate(test_loader):
        input_, label = data_
        val_input = Variable(input_).to(available_device)
        val_label = Variable(label.type(torch.LongTensor)).to(available_device)
        score = C.forward_partial(val_input)
        y_pred_batch = score.detach().cpu().squeeze().numpy()
        y_real_batch = val_label.cpu().data.squeeze().numpy()
        y_real.append(y_real_batch.tolist())
        y_pred.append(y_pred_batch.tolist())

    y_real = [item for batch in y_real for item in batch]
    y_pred = [item for batch in y_pred for item in batch]
    
    return y_real, y_pred

def iterations_test_partial_image(C, test_loader, available_device = "cpu"):
    y_real = list()
    y_pred = list()
    images = list()
    for ii, data_ in enumerate(test_loader):
        input_, label = data_
        val_input = Variable(input_).to(available_device)
        val_label = Variable(label.type(torch.LongTensor)).to(available_device)
        score = C.forward_partial(val_input)
        y_pred_batch = score.detach().cpu().squeeze().numpy()
        y_real_batch = val_label.cpu().data.squeeze().numpy()
        y_real.append(y_real_batch.tolist())
        y_pred.append(y_pred_batch.tolist())
        images.append(val_input.detach().cpu().squeeze().numpy().tolist())

    y_real = [item for batch in y_real for item in batch]
    y_pred = [item for batch in y_pred for item in batch]
    images = [item for batch in images for item in batch]
    
    return y_real, y_pred, images


if __name__ == "__main__":
    pass