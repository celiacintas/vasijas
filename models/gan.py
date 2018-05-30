import os
import torch
import time
import pickle
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms as tfs
from torch.utils import data
from models.generator import _G
from models.discriminator import _D
import utils
import numpy as np

class GAN(object):
    """
    See https://github.com/znxlwm/pytorch-generative-model-collections
    for more info on GANs
    """
    def __init__(self, epochs=100, sample=25, batch=64, 
                 input_h_w=112, latent_v=64, transforms=None):
        # parameters
        self.epoch = epochs
        self.sample_num = sample
        self.batch_size = batch
        self.save_dir = '/tmp/'
        self.result_dir = '/tmp/'
        self.log_dir = '/tmp/'
        self.gpu_mode = True
        self.dataset = 'vasijas'
        self.model_name = 'GAN'
        
        # networks init
        self.G = _G(input_h_w, latent_v)
        self.D = _D(input_h_w)
        self.G_optimizer = optim.Adam(self.G.parameters(),
                                      lr=0.0002, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(),
                                      lr=0.0002, betas=(0.5, 0.999))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)

        # load dataset
        imagenet_data = datasets.ImageFolder('data/png/',
                                             tfs.Compose(
                                             transforms))

        self.data_loader = data.DataLoader(imagenet_data,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.z_dim = latent_v

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size,
                                                 self.z_dim)).cuda(),
                                                 volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size,
                                                  self.z_dim)),
                                                  volatile=True)
            
    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        output_path = '/'.join([self.result_dir, self.dataset,
                               self.model_name])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, 
                                     self.z_dim)).cuda()) #, volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size,
                                     self.z_dim))) #, volatile=True)

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter,  (x_, _) in enumerate(self.data_loader):
                # print("esta es la forma de mi batch,", x_.shape)
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        animation_path = '/'.join([self.result_dir, self.dataset,
                                  self.model_name, self.model_name])
        utils.generate_animation(animation_path, self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
