from train import train
from test import test
from utils import ARR2TXT
from dataset import standardDate
import numpy as np
import os

def run(type_conf, common_conf, train_conf, test_conf, input_conf, ftimes):
    vars       = input_conf['vars']
    utcs       = input_conf['utcs']
    gis        = np.load(input_conf['gis_dir'])
    mask       = np.load(input_conf['mask_dir'])
    if type_conf == 'train':
        for var in vars:
            for utc in utcs:
                for ftime in ftimes:
                    ftime = str(ftime)
                    train(common_conf, train_conf, input_conf, var, utc, ftime)
    elif type_conf == 'test':
        print('DATA making with SRGAN start')
        for var in vars:
            for utc in utcs:
                datelist = standardDate(input_conf['sdate']+utc, input_conf['edate']+utc)
                for date in datelist:
                    save_file = np.zeros((745, 1265, 28))
                    for ftime in ftimes:
                        ftime       = ftime
                        time_output = test(var, utc, date, str(ftime), gis, mask, common_conf, 
                                           test_conf, input_conf)
                        save_file[:,:,ftime-2] = time_output
                    ppm = ARR2TXT(save_file, date)
                    save_name = 'DFS_SHRT_STD_GDPS_SRGN_%s'%(var.upper())
                    ppm.toTXT(test_conf['output_dir'] , save_name)
                    os.chmod('%s/%s.%s'%(test_conf['output_dir'], save_name, date), 0o755)
                    print('%s %s is saved' %(save_name, str(date)))
