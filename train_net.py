
import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
import utils.network_vox as nv
import click
import sh

Z_LATENT_SPACE = 128
Z_INTERN_SPACE = 136
CUBE_LEN = 64

G_LR = 0.00002
D_LR = 0.0002
EPOCHS = 100
BSIZE = 32
BETAS = (0.9, 0.999)


@click.command()
@click.option('--nepoch', default=EPOCHS, help='30, 40, 50, 60')
@click.option('--bsize', default=BSIZE, help='16, 32, 64')
@click.option('--lrG', default=G_LR, help='0.00001, 0.0001, 0.001')
@click.option('--lrD', default=D_LR, help='0.00001, 0.0001, 0.001')
@click.option('--dirCheckpoint', default="weight", help='')
@click.option('--dirDataset', default="./data/", help='')
@click.option('--available_device', default="cuda", help='cuda, cuda:0, cuda:1, cpu')
def main(nepoch, bsize, lrg, lrd, dircheckpoint, dirdataset, available_device):
    print(20*"-")
    print('n_epoch:', nepoch)
    print('batch size:', bsize)
    print("lrG:", lrg)
    print("lrD:", lrd)
    print("dircheckpoint:", dircheckpoint)
    print("dirDataset:", dirdataset)
    print("device:", available_device)
    print(20*"-")

    sh.mkdir("-p", dircheckpoint)

    dt = FragmentDataset(dirdataset, 'train')

    D = nv._D().to(available_device)
    G_encode_decode = G_encode_decode = nv._G_encode_decode(
        cube_len=CUBE_LEN, z_latent_space=Z_LATENT_SPACE, z_intern_space=Z_INTERN_SPACE).to(available_device)

    G_encode_decode_optimizer = optim.Adam(G_encode_decode.parameters(),
                                           lr=lrg, )
    D_optimizer = optim.Adam(D.parameters(),
                             lr=lrd, )

    data_loader = data.DataLoader(
        dt, batch_size=bsize, shuffle=True, drop_last=True)

    crit_D = nn.BCELoss()
    crit_G = nn.BCELoss()

    for epoch in range(EPOCHS):
        for i,  (mesh_frag, mesh_complete) in enumerate(data_loader):

            G_encode_decode.zero_grad()

            mesh_frag = mesh_frag.float().to(available_device)
            mesh_complete = mesh_complete.float().to(available_device)

            y_real_ = torch.tensor(np.random.uniform(
                0.8, 1.0, (bsize))).to(available_device).float()
            y_fake_ = torch.tensor(np.random.uniform(
                0, 0.20, (bsize))).to(available_device).float()

            # update G network
            output_g_encode = G_encode_decode.forward_encode(mesh_frag)
            fake = G_encode_decode.forward_decode(output_g_encode)

            fake = fake + (mesh_frag.unsqueeze(1))

            D_fake = D(fake).view(bsize)
            G_loss = crit_G(D_fake, y_real_)

            G_loss.backward()
            G_encode_decode_optimizer.step()

            # update D network
            D.zero_grad()

            output_g_encode = G_encode_decode.forward_encode(mesh_frag)

            fake = G_encode_decode.forward_decode(output_g_encode)

            fake = fake + (mesh_frag.unsqueeze(1))

            D_fake = D(fake).view(bsize)
            D_fake_loss = crit_D(D_fake, y_fake_)

            D_real = D(mesh_complete).view(bsize)
            D_real_loss = crit_D(D_real, y_real_)

            D_loss = (D_real_loss + D_fake_loss) / 2

            D_loss.backward()
            D_optimizer.step()

            print("Epoch: [%2d / %2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                  ((epoch + 1), EPOCHS, (i + 1), data_loader.dataset.__len__() //
                   BSIZE, D_loss.item(), G_loss.item()))

        if (epoch+1) % 15 == 0:
            torch.save(G_encode_decode.state_dict(
            ), '{}/G_encode_decode_partial_{}.pkl'.format(dircheckpoint, epoch+1))
            torch.save(D.state_dict(),
                       '{}/D_partial_{}.pkl'.format(dircheckpoint, epoch+1))

    torch.save(G_encode_decode.state_dict(),
               '{}/G_encode_decode_final_{}.pkl'.format(dircheckpoint, epoch))
    torch.save(D.state_dict(),
               '{}/D_final_{}.pkl'.format(dircheckpoint, epoch))


if __name__ == '__main__':
    main()
