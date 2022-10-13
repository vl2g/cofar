import json
import torch
# import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import errno
import os
import os.path as op
import yaml
import random
import numpy as np
from omegaconf import OmegaConf

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

# class MetricTracker:
#     def __init__(self, *keys, writer=None):
#         self.writer = writer
#         self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
#         self.reset()

#     def reset(self):
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key, value)
#         self._data.total[key] += value * n
#         self._data.counts[key] += n
#         self._data.average[key] = self._data.total[key] / self._data.counts[key]

#     def avg(self, key):
#         return self._data.average[key]

#     def result(self):
#         return dict(self._data.average)
