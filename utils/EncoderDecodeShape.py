import torch
import torch.nn as nn
import numpy as np


class EncoderDecodeShape(torch.nn.Module):
    def __init__(self, nc=1, ndf=128, nz=128, ngf=128, dropout_rate = 0.5 ):
        super(EncoderDecodeShape, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.05),
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            ##nn.Conv2d(1, 1, 5, 1, 0, bias=False),
            ##nn.Sigmoid()
        )
        
        self.linearEncoder = nn.Sequential(
            nn.Linear(64+32, 128)
        )
        
        
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.Dropout(0.05),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            # state size. (ngf*8) x 4 x 4 == 1024 x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),            # state size. (ngf*4) x 8 x 8  == 512 x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),            # state size. (ngf*2) x 16 x 16 == 256 x 4 x 4
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),            # state size. (ngf) x 32 x 32 == 128 x 4 x 4
            nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )


    def join_tope_base(self, v_tope, v_base):
        v = torch.cat((v_tope, v_base), 1).reshape(-1, 2, 32)
        
        return v.max(1)[0]
    
    def forward(self, x, v_tope, v_base):
        encoded = self.forward_encoder(x, v_tope, v_base)
        decoded = self.forward_decoder(encoded)
        return decoded
    
    def forward_encoder(self, x, v_tope, v_base):
        
        encoded = self.encoder(x).reshape(-1, 64)
        tope_base = self.join_tope_base(v_tope, v_base)
        
        x_out = torch.cat((encoded, tope_base), 1)
        
        return self.linearEncoder(x_out).unsqueeze(2).unsqueeze(2)
    
    def forward_decoder(self, encoded):
        decoded = self.decoder(encoded)
        return decoded



