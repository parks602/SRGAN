import argparse
import os
import math
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from dataset import DatasetMaker, MinMaxscaler
from loss import GeneratorLoss
from model import Generator, Discriminator
from utils import EarlyStopping
from torch.cuda.amp import GradScaler, autocast
import numpy as np


def train(common_conf, train_conf, input_conf, var, utc, ftime):
    print('START %s %s %s' %(var, utc, ftime.zfill(2)))

    USE_CUDA     = torch.cuda.is_available()
    device       = torch.device('cuda' if USE_CUDA else 'cpu')
    print('Train Device : ', device)
    model_name        = train_conf['model_name']%(var, str(int(ftime)*3).zfill(2), utc, 
                                                  input_conf['sdate'], input_conf['edate'])
    model_save_dir    = '%s/%s'%(common_conf['model_path'], model_name)

    if not os.path.exists(common_conf['model_path']):
        os.makedirs(common_conf['model_path'])

    early_stopping    = EarlyStopping(patience=train_conf['patience'], verbose=True, path= model_save_dir)

    UPSCALE_FACTOR    = common_conf['upscale_factor']
    NUM_EPOCHS        = train_conf['epoch']

    ################################################################################
    '''MAKE DATASET'''  
    ################################################################################
    mask              = np.load(input_conf['mask_dir'])
    gis               = np.load(input_conf['gis_dir'])
    gis               = MinMaxscaler(0, 2600, gis)
    hight, landsea    = [], []

    for ii in range(train_conf['batch_size']):
        hight.append(gis)
        landsea.append(mask)

    hight, landsea    = np.asarray(hight), np.asarray(landsea)

    hight             = np.expand_dims(hight, axis=1)
    landsea           = np.expand_dims(landsea, axis=1)

    hight             = torch.as_tensor(hight, dtype=torch.float)
    landsea           = torch.as_tensor(landsea, dtype=torch.float)

    real_hight        = Variable(hight).to(device)
    real_landsea      = Variable(landsea).to(device)

    fake_label        = torch.full((train_conf['batch_size'], 1), 0, dtype=hight.dtype).to(device)
    real_label        = torch.full((train_conf['batch_size'], 1), 1, dtype=hight.dtype).to(device)

    train_loader, val_loader = DatasetMaker(input_conf, train_conf, var, utc, ftime)


    ################################################################################
    '''MAKE MODEL'''  
    ################################################################################
    netG              = Generator(UPSCALE_FACTOR)
    netD              = Discriminator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion     = GeneratorLoss()
    discriminator_criterion = nn.BCEWithLogitsLoss().to(device)

    scalerD           = GradScaler()
    scalerG           = GradScaler()

    netG.to(device)
    netD.to(device)
    generator_criterion.to(device)
    discriminator_criterion.to(device)

    optimizerG        = optim.Adam(netG.parameters())
    optimizerD        = optim.Adam(netD.parameters())

    schedulerG        = optim.lr_scheduler.OneCycleLR(optimizerG,
                                                      max_lr = 0.001,
                                                      pct_start = 0.1,
                                                      epochs = NUM_EPOCHS,
                                                      steps_per_epoch = len(train_loader),\
                                                      anneal_strategy='linear')
    schedulerD        = optim.lr_scheduler.OneCycleLR(optimizerD,
                                                      max_lr = 0.001,
                                                      pct_start = 0.1,
                                                      epochs = NUM_EPOCHS,
                                                      steps_per_epoch = len(train_loader),
                                                      anneal_strategy='linear')

    results            = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'val_loss': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size != train_conf['batch_size']:
                continue

            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img         = Variable(target).to(device)
            real_img         = real_img.to(device)
            z                = Variable(data).to(device)

            fake_img = netG(z, real_landsea, real_hight)

            netD.zero_grad()
            with autocast():
                real_out     = discriminator_criterion(netD(real_img), real_label)
                fake_out2    = discriminator_criterion(netD(fake_img.detach()), fake_label)
                d_loss       = real_out + fake_out2

            scalerD.scale(d_loss).backward(retain_graph=True)
            scalerD.step(optimizerD)
            scalerD.update()
            schedulerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            with autocast():
                fake_img     = netG(z, real_landsea, real_hight)
                fake_out     = netD(fake_img).mean()
                g_loss       = generator_criterion(fake_out, fake_img, real_img)

            scalerG.scale(g_loss).backward()
            scalerG.step(optimizerG)
            scalerG.update()
            schedulerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            train_bar.set_description(desc='[%d/%d] Loss_D:%.2f Loss_G:%.2f' % (
                epoch, NUM_EPOCHS,
                running_results['d_loss'] / running_results['batch_sizes'],
                math.sqrt(running_results['g_loss'] / running_results['batch_sizes']),
                ))

        netG.eval()

        if epoch%2 == 1:
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'val_loss': 0, 'batch_sizes': 0}
                val_images = []
                for val_data, val_target in val_bar:
                    batch_size = val_data.size(0)
                    if batch_size != train_conf['batch_size']:
                        continue
                    valing_results['batch_sizes'] += batch_size
                    lr = val_data.to(device)
                    hr = val_target.to(device)
                    sr  = netG(lr, real_landsea, real_hight)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse.item() * batch_size
                    val_bar.set_description(
                        desc='[converting LR images to SR images] Val_loss: %.4f' % (
                            math.sqrt(valing_results['mse'] /  valing_results['batch_sizes'])))

        # save model parameters
            early_stopping(math.sqrt(valing_results['mse']/ valing_results['batch_sizes']), netG)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['val_loss'].append(valing_results['mse']/ valing_results['batch_sizes'])
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                  'Val_loss': results['val_loss'] },
            index=range(1, epoch + 1))
        #data_frame.to_csv('%s/train_results.csv'%(save_dir), index_label='Epoch')

