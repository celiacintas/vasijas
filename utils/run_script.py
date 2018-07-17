import os
import torch
from torch import optim
from torch import nn
from utils import utils3D
from torch.utils import data
from torch.autograd import Variable
from models.threed.gan import GAN
import matplotlib
import pickle

Z_LATENT_SPACE = 200
G_LR = 0.0025
D_LR = 0.001
EPOCHS = 500
BETA = (0.5, 0.5) 
BSIZE = 32
CUBE_LEN = 64

def main():
    gan3D = GAN(epochs=EPOCHS, sample=8, 
            batch=BSIZE, betas=BETA,
            g_lr=G_LR, d_lr=D_LR, cube_len=CUBE_LEN, latent_v=Z_LATENT_SPACE)
    gan3D.train()
    gan3D.save()

if __name__ == '__main__':
    main()
    
