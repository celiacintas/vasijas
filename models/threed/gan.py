import os
import time
import torch
from torch import optim
from torch import nn
from utils import utils3D, utils
from torch.utils import data
from torch.autograd import Variable
from models.threed.generator import _G
from models.threed.discriminator import _D
import pickle
import numpy as np


class GAN(object):
    """
    See https://github.com/meetshah1995/tf-3dgan and https://github.com/rimchang/3DGAN-Pytorch 
    for more info on 3DGANs
    """
    def __init__(self, epochs=100, sample=25, batch=32, betas=(0.5, 0.5),
                 g_lr=0.0025, d_lr= 0.001, cube_len=64, latent_v=200,
                 data_path='output/output_obj/', transforms=None):
        # parameters
        self.epoch = epochs
        self.betas = betas
        self.sample_num = sample
        self.batch_size = batch
        self.save_dir = '/tmp/'
        self.result_dir = '/tmp/'
        self.log_dir = '/tmp/'
        self.gpu_mode = True
        self.dataset = 'vasijas'
        self.model_name = 'GAN3D'
        
        # networks init
        self.G = _G(z_latent_space=latent_v)
        self.D = _D()
        self.G_optimizer = optim.Adam(self.G.parameters(),
                                      lr=g_lr, betas=self.betas)
        self.D_optimizer = optim.Adam(self.D.parameters(),
                                      lr=d_lr, betas=self.betas)

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
        imagenet_data = utils3D.VesselsDataset(data_path)

        self.data_loader =  data.DataLoader(imagenet_data, 
                                            batch_size=self.batch_size,
                                            shuffle=True, num_workers=1)
        self.z_dim = latent_v

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size,
                                                 self.z_dim)).cuda())
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size,
                                                  self.z_dim)))
            
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

    def visualize_results(self, samples, epoch):
        output_path = '/'.join([self.result_dir, self.dataset,
                               self.model_name])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        utils3D.save_plot_voxels(samples, output_path, epoch)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []


        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            print('epoch nro {}'.format(epoch))
            epoch_start_time = time.time()
            for i,  X in enumerate(self.data_loader):
                x_ = utils3D.var_or_cuda(X)      
                print("Batch nro {}".format(i))    
                if x_.size()[0] != int(self.batch_size):
                    print("batch_size != {} drop last incompatible batch".format(int(self.batch_size)))
                    continue

                z_ = utils3D.var_or_cuda(torch.randn(self.batch_size, self.z_dim))
                self.y_real_, self.y_fake_ = utils3D.var_or_cuda(torch.ones(self.batch_size)), \
                                             utils3D.var_or_cuda(torch.zeros(self.batch_size))

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])


                d_real_acu = torch.ge(D_real_loss.squeeze(), 0.5).float()
                d_fake_acu = torch.le(D_fake_loss.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

                if d_total_acu.data[0] <= 0.8:
                    self.D.zero_grad()
                    D_loss.backward()
                    self.D_optimizer.step()

                # update G network
                z_ = utils3D.var_or_cuda(torch.randn(self.batch_size, self.z_dim))   

                #self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.data[0])

                self.D.zero_grad()
                self.G.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                if ((epoch + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (epoch + 1), self.data_loader.dataset.__len__() // 
                          self.batch_size, D_loss.data[0], G_loss.data[0]))
                    self.visualize_results((epoch+1))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            samples = G_.cpu().data[:self.sample_num].squeeze().numpy()
            self.visualize_results(samples, (epoch+1))
            self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        #animation_path = '/'.join([self.result_dir, self.dataset,
        #                          self.model_name, self.model_name])
        #if not os.path.exists(animation_path):
        #    os.makedirs(animation_path)
        #utils.generate_animation(animation_path, self.epoch)
        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
