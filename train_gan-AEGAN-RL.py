from torchvision import transforms as tfs
from torch.utils import data
import PIL.ImageOps
from utils import CustomDataGAN as cdata
import numpy as np
from utils import utils
import os
import pickle
from utils import Discriminator as netD
from utils import GeneratorV8 as netG
import torch.nn as nn
import torch.optim as optim
import torch
from utils.Plotter import VisdomLinePlotter
import tqdm

available_device = 'cuda'
envplotter = 'gan-v8'
plotter = VisdomLinePlotter(port=8097, env_name=envplotter)

transformations = [
                   tfs.Grayscale(),
                   #tfs.RandomHorizontalFlip(p=0.7), 
                   #tfs.RandomAffine(0, scale=(0.7, 1.), fillcolor=(255,)), 
                   tfs.RandomRotation(degrees=(0,45), fill=(255,)), 
                   tfs.Resize((128, 128)),
                   
                   tfs.Lambda(lambda x: PIL.ImageOps.invert(x)),
                   tfs.ToTensor()
]

transformations_target = [
                   tfs.Resize((128, 128)),
                   tfs.Grayscale(),
                   tfs.Lambda(lambda x: PIL.ImageOps.invert(x)),
                   tfs.ToTensor()
]

imagenet_data = cdata.CustomDataGAN(
                            '../vasijas/data/perfiles_CATA/png_clasificados/',
                            '../fragmentos_vasijas/morfo/',
                            transform=tfs.Compose(transformations),
                            transform_target = tfs.Compose(transformations_target))


BATCH_SIZE = 4

splits_len = round(len(imagenet_data.samples)*0.2), round(len(imagenet_data.samples)*0.1), round(len(imagenet_data.samples)*0.7)+1
splits_len, np.sum(splits_len), len(imagenet_data.samples) 
#### Random split
splits = utils.random_split(imagenet_data, splits_len)
train_loader = data.DataLoader(splits[2], batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(splits[1], batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(splits[0], batch_size=BATCH_SIZE, shuffle=True)

if not os.path.exists('datapickle/gan_0.pickle'):
        print('Creating indexs files')
    
        filehandler = open("datapickle/gan_0.pickle", 'wb') 
        pickle.dump(splits[0].indices, filehandler)

        filehandler = open("datapickle/gan_1.pickle", 'wb') 
        pickle.dump(splits[1].indices, filehandler)

        filehandler = open("datapickle/gan_2.pickle", 'wb') 
        pickle.dump(splits[2].indices, filehandler)
        
else:
        print('Load indexs files')
        
        file_pi2 = open('datapickle/gan_0.pickle', 'rb') 
        splits[0].indices = pickle.load(file_pi2)
        
        file_pi2 = open('datapickle/gan_1.pickle', 'rb') 
        splits[1].indices = pickle.load(file_pi2)

        file_pi2 = open('datapickle/gan_2.pickle', 'rb') 
        splits[2].indices = pickle.load(file_pi2)

LR_G = 0.00002 
LR_D = 0.000002

# Model Initialization
model_G = netG.Generator(nc_input=1, nc_output=1).to(available_device)
#checkpoint = torch.load("models/generador_v8_5000.pkl")
#model_G.load_state_dict(checkpoint)


model_D = netD.Discriminator(nc=3).to(available_device)
#checkpoint = torch.load("models/discriminador_v8_5000.pkl")
#model_D.load_state_dict(checkpoint)



optim_G = optim.Adam(model_G.parameters(), lr=LR_G)
optim_D = optim.Adam(model_D.parameters(), lr=LR_D)

crit_D = nn.BCELoss()
crit_G = nn.BCELoss()

# setup optimizer
epochs = 5005
for epoch in tqdm.tqdm(range(epochs)):
    loss_batch_G = []
    loss_batch_D = []

    for batch, (image, labels, ids_name, tope, base) in enumerate(train_loader, 0):

        valid = torch.tensor(np.random.uniform(0.7, 1.2, (image.size(0), 1))).to(available_device).float()
        fake = torch.tensor(np.random.uniform(0, 0.33, (image.size(0), 1))).to(available_device).float()

        optim_G.zero_grad()
        tope, base, image = tope.to(available_device), base.to(available_device), image.to(available_device)
        target = torch.cat((tope, base, image), 1)
        #origin = torch.cat((tope, base), 1)

        predicted_r = model_G(tope, base)
        predicted = torch.cat((tope, base, predicted_r), 1)
        
        
        loss_G =  crit_G(model_D(predicted).reshape(-1, 1), valid)
        
        #loss_batch_face.append(loss_G_face)
        loss_batch_G.append(loss_G.item())

        loss_G.backward()
        optim_G.step()

        # TRAIN D   
        optim_D.zero_grad()
        predicted_r = model_G(tope, base)
        predicted = torch.cat((tope, base, predicted_r), 1)

        real_loss = crit_D(model_D(target).reshape(-1, 1), valid) 
        fake_loss = crit_D(model_D(predicted).reshape(-1, 1), fake)  
        loss_D = ((real_loss + fake_loss) / 2 )

        ##loss_D = crit_D(model_D(target).reshape(-1, 1), model_D(predicted).reshape(-1, 1)) 

        loss_D.backward()
        optim_D.step()
        loss_batch_D.append(loss_D.item())

    plotter.plot("Loss", "G", epoch, np.mean(loss_batch_G)) 
    plotter.plot("Loss", "D", epoch, np.mean(loss_batch_D))
    
    predicted_show = torch.cat((tope, base, predicted_r, image), 1)
    plotter.images(predicted_show.reshape(-1, 1, 128, 128), nrow=4)
    
    torch.save(model_G.state_dict(), 'models/generador_v8_current_{}.pkl'.format(epochs))
    torch.save(model_D.state_dict(), 'models/discriminador_v8_current_{}.pkl'.format(epochs))
    
    for batch, (image, labels, ids_name, tope, base) in enumerate(val_loader, 0):
        
        tope, base, image = tope.to(available_device), base.to(available_device), image.to(available_device)
        target = torch.cat((tope, base, image), 1)
        #origin = torch.cat((tope, base, (tope+base).reshape(base.shape)), 1)

        predicted_r = model_G(tope, base)
        predicted_show = torch.cat((tope, base, predicted_r, image), 1)
        plotter.images_val(predicted_show.reshape(-1, 1, 128, 128), nrow=4, title='Images VAL')
        break

torch.save(model_G.state_dict(), 'models/generador_v8_{}.pkl'.format(epochs))
torch.save(model_D.state_dict(), 'models/discriminador_v8_{}.pkl'.format(epochs))


