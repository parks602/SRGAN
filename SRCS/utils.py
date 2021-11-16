import sys, os, json, csv
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime, timedelta
from torch.utils.data import Dataset

def readConfig(fconfig):
    with open(fconfig,'r') as f:
        confg = json.load(f)
        type_conf  = confg['type']
        comm_conf  = confg['common']
        train_conf = confg['train']
        test_conf  = confg['test']
        input_conf = confg['input']
        return type_conf, comm_conf, train_conf, test_conf, input_conf


def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Make Directories : %s" %(path))


class ARR2TXT:
    def __init__(self, dt, anltim):
        """
         dt    : 3d data (nlon, nlat, nftim)
         anltim: yyyymmddhh (str)
         nt    : forecast timeseries (interval : 3h)
        """
        self.anltim = anltim
        self.nx     = dt.shape[0]
        self.ny     = dt.shape[1]
        self.nt     = dt.shape[2]
    
        if self.nt > 28:
            pad = self.nt - 28
            dt = dt[:,:,pad:] ## start 6h ~ 87h
        self.data   = dt
  
    def _LocalTime(self, ftim):
        fmt = '%Y%m%d%H'
        lst = datetime.strptime(self.anltim,fmt) + timedelta(hours=9) + timedelta(hours=ftim)
        return datetime.strftime(lst,fmt)
  
    def _output_header(self, utc):
        lst = self._LocalTime(utc)
        line = '%s+%03dHOUR     %sLST\n' %(self.anltim, utc, lst)
        return line
  
    def toTXT(self, opath, prefix, o_in_line=10):
        MakeDir(opath)
        ofile = '%s/%s.%s' %(opath, prefix, self.anltim)
        with open(ofile, 'w') as f:
            for i in range(self.nt):
                temp = self.data[:,:,i].flatten()
                q = int(temp.size / o_in_line) + 1
                utc = i * 3 + 6
                f.write(self._output_header(utc))
                for j in range(q):
                    idx = o_in_line * j
                    eidx = o_in_line * (j+1)
                    if temp.size < eidx: eidx = temp.size
                    for k in range(o_in_line):
                        if eidx <= idx + k: break
                        f.write('%7.1f' %(temp[idx+k:idx+k+1]) )
                    if j != (q-1): f.write('\n')
                if (temp.size % o_in_line) != 0 and i != (self.nt - 1): f.write('\n')


class ReadASCII:
    def __init__(self, fname, nline=10):
        if not os.path.exists(fname):
            print("Can't find %s" %(fname))
            self.data = None
        else:
            data = []
            with open(fname,'r') as f:
                ### Read Header
                nx, ny = list(map(int,f.readline().strip().split(",")))

                reader = csv.reader(f)
                for row in reader:
                    data.extend(list(map(float,row[0].split())))
            self.data = np.reshape(np.array(data).flatten(),[nx,ny])

    def getData(self):
        return self.data

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

