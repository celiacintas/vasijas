import torch.nn as nn
import torch

class Generator(torch.nn.Module):
    def __init__(self, nc_input=1, nc_output=1, ndf=128, nz=128, ngf=128, dropout_rate = 0.5 ):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.05),
            # input is (nc) x 64 x 64
            nn.Conv2d(nc_input, ndf, 4, 2, 1, bias=False),
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
            nn.Linear(64, 128)
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
            nn.ConvTranspose2d(    ngf,      nc_output, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def join_tope_base(self, v_tope, v_base):
        v = torch.cat((v_tope, v_base), 1).reshape(-1, 2, 128)
        
        return v.max(1)[0]

    def forward(self, x_base, x_tope, ):
        encoded_base = self.forward_encoder(x_base)
        encoded_tope = self.forward_encoder(x_tope)
        x = self.join_tope_base(encoded_base, encoded_tope)
        decoded = self.forward_decoder(x.reshape(-1, 128, 1, 1))
        return decoded
    
    def forward_encoder(self, x):
        encoded = self.encoder(x).reshape(-1, 64)
        return self.linearEncoder(encoded).unsqueeze(2).unsqueeze(2)
    
    def forward_decoder(self, encoded):
        decoded = self.decoder(encoded)
        return decoded


