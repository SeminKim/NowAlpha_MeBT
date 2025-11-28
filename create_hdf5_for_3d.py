import importlib
import numpy as np
import torch
from taming.data.hsr import HSRDebug
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

dataset = HSRDebug('your_data_list.txt', pooling=(1024, 1024), spatial_to_channel=(4, 4))

dataloader = DataLoader(dataset, 8, num_workers=8, persistent_workers=True)

f = h5py.File('your_h5_file.h5', 'w')

with torch.no_grad():
    for i, b in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i == 0:
            f.create_dataset('datestr', data=np.array(b['datestr'], dtype='S'), compression="gzip", chunks=True, maxshape=(None, ))
            f.create_dataset('check', data=b['return'].numpy(), compression="gzip", chunks=True, maxshape=(None, ))
        else:
            B_orig = len(f['datestr'])
            B = len(b['datestr'])

            f['datestr'].resize(B_orig + B, axis=0)
            f['datestr'][-B:] = np.array(b['datestr'], dtype='S')

            f['check'].resize(B_orig + B, axis=0)
            f['check'][-B:] = b['return'].numpy()

f.close()

print('Done')
