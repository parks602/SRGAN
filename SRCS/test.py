import argparse, os
import numpy as np
import pandas as pd
import torch
import dataset
from datetime import datetime, timedelta
from torch.autograd import Variable
from model import Generator
from torch.utils.data import DataLoader


def test(var, utc, date, ftime, gis, mask, common_conf, test_conf, input_conf):
    USE_CUDA           = torch.cuda.is_available()
    device             = torch.device('cuda' if USE_CUDA else 'cpu')

    input              = dataset.test_Dataset(input_conf, var, date, ftime)
    test_dataset       = dataset.test_datasets3d(input)
    test_loader        = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers=4)
  
    gis                = dataset.MinMaxscaler(0, 2600, gis)
    hight, landsea     = [], []
  
    for ii in range(test_conf['batch_size']):
        hight.append(gis)
        landsea.append(mask)
  
    hight, landsea     = np.asarray(hight), np.asarray(landsea)
    hight              = np.expand_dims(hight, axis=1)
    landsea            = np.expand_dims(landsea, axis=1)
  
    hight              = torch.as_tensor(hight, dtype=torch.float)
    landsea            = torch.as_tensor(landsea, dtype=torch.float)
  
    real_hight         = Variable(hight).to(device)
    real_landsea       = Variable(landsea).to(device)
  
    #best_loc = notebook['PSNR'].idxmax()
    model              = Generator(common_conf['upscale_factor']).eval()
    model.to(device)
    MODEL_NAME         = test_conf['model_name']%(var.lower(), str(int(ftime)*3).zfill(2), utc, 
                                                  test_conf['model_sdate'], test_conf['model_edate'])

    model.load_state_dict(torch.load('%s%s' %(common_conf['model_path'], MODEL_NAME), map_location=device))

    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.to(device)
            input = Variable(input)
            output = model(input, real_landsea, real_hight)
            output = output[0][0]
            if USE_CUDA:
                output = output.cpu()
            out = output.clone().detach().numpy()
  
    return out
