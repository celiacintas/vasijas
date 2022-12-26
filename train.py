
import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
import utils.network_vox as nv

Z_LATENT_SPACE = 128
G_LR = 0.00002 
D_LR = 0.0002
EPOCHS = 100
BSIZE = 32
CUBE_LEN = 64
BETAS = (0.9, 0.999)


dt = FragmentDataset('./data', 'train')

available_device = "cuda" if torch.cuda.is_available() else "cpu"

D = nv._D().to(available_device)
G_encode_decode = nv._G_encode_decode(z_latent_space=Z_LATENT_SPACE).to(available_device)

G_encode_decode_optimizer = optim.Adam(G_encode_decode.parameters(),
                                      lr=G_LR, )
D_optimizer = optim.Adam(D.parameters(),
                                      lr=D_LR, )

data_loader = data.DataLoader(dt, batch_size=BSIZE, shuffle=True, drop_last=True)

crit_D = nn.BCELoss()
crit_G = nn.BCELoss()

for epoch in range(EPOCHS):
    for i,  (mesh_frag, mesh_complete) in enumerate(data_loader):
        
        G_encode_decode.zero_grad()
        
        #print("Batch nro {}".format(i))

        mesh_frag = mesh_frag.float().to(available_device)
        mesh_complete = mesh_complete.float().to(available_device)

        y_real_ = torch.tensor(np.random.uniform(0.8, 1.0, (BSIZE))).to(available_device).float()
        y_fake_ = torch.tensor(np.random.uniform(0, 0.20, (BSIZE))).to(available_device).float()
        
        
        # update G network
        output_g_encode = G_encode_decode.forward_encode(mesh_frag)
        fake = G_encode_decode.forward_decode(output_g_encode) 
        
        fake = fake + (mesh_frag.unsqueeze(1))
        
        D_fake = D(fake).view(BSIZE)
        G_loss = crit_G(D_fake, y_real_)

    
        G_loss.backward()
        G_encode_decode_optimizer.step()
        
        # update D network
        D.zero_grad()

        output_g_encode = G_encode_decode.forward_encode(mesh_frag)

        fake = G_encode_decode.forward_decode(output_g_encode)
        
        fake = fake + (mesh_frag.unsqueeze(1))
        
        D_fake = D(fake).view(BSIZE)
        D_fake_loss = crit_D(D_fake, y_fake_)
        
        D_real = D(mesh_complete).view(BSIZE)
        D_real_loss = crit_D(D_real, y_real_)

        D_loss = (D_real_loss + D_fake_loss) / 2
        
        
        D_loss.backward()
        D_optimizer.step()

        print("Epoch: [%2d / %2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                    ((epoch + 1),EPOCHS, (i + 1), data_loader.dataset.__len__() // 
                    BSIZE, D_loss.item(), G_loss.item()))
        

    if (epoch+1) % 15 == 0:
        torch.save(G_encode_decode.state_dict(), 'weight/v2_G_encode_decode_partial_{}.pkl'.format(epoch+1))
        torch.save(D.state_dict(), 'weight/v2_D_partial_{}.pkl'.format(epoch+1))

    
torch.save(G_encode_decode.state_dict(), 'weight/v2_G_encode_decode_final_{}.pkl'.format(epoch))
torch.save(D.state_dict(), 'weight/v2_D_final_{}.pkl'.format(epoch))