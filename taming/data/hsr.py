import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
### For Satellite Dataset
import gzip
from tqdm import tqdm
from taming.data.utils import spatial_to_channel as s2c
INT_MAX = 2**31

INVALID = -1e-3

def zr_relation(x):
    x = 10**(x*0.1)
    return (x/148.)**(100./159.)

def normalize_zr(x):
    return torch.tanh(torch.log(x+0.01)/4.)

def unnormalize_zr(x):
    # return value: -0.01 ~ 4M
    return torch.exp(4*torch.arctanh(x.clamp(-0.999, 0.999)))-0.01

class HSRBase(Dataset):
    def __init__(self, data_list:str, wrong_idx_txt:str, pooling=None, spatial_to_channel=None):
        super().__init__()
        self.data = []
        self.spatial_to_channel = spatial_to_channel
        # hsr params
        self.pooling = False if not pooling else tuple(pooling)
        self.seed = np.random.randint(65536)

        with open(data_list, "r") as f:
            paths = f.read().splitlines()
        print(f"# of Raw Data: {len(paths)}")

        # find wrong idx
        with open(wrong_idx_txt, 'r') as f_wrong:
            self.wrong_idx = f_wrong.read().splitlines()
        for f_name in tqdm(paths, desc='Data Preparing'):
            if any(subname in f_name for subname in self.wrong_idx):
                continue
            # # Data Checking
            # # I checked this once, so no need to do it again.
            # try:
            #     with gzip.open(f_name, 'rb') as f:
            #         pass
            # except:
            #     print('Error occured at dataset initialization. Please add {f_name} to {wrong_idx_txt}.')
            self.data.append(f_name)
        print(f"# of Prepared Data: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Simple fix for dirty dataset
        self.seed += i
        self.seed = self.seed % INT_MAX
        rng = np.random.default_rng(self.seed)
        i = rng.integers(len(self.data))
        try:
            target = self.parse_hsr(self.data[i])        
            if self.spatial_to_channel is not None:
                target = s2c(target, self.spatial_to_channel)
            mask = target < 0.
            target = torch.where(mask, INVALID * torch.ones_like(target), target)
            return {"image": target, "mask": mask}
        except:
            print(f'Something went wrong with parsing data[{i}]:\n{self.data[i]}')
            return self.__getitem__(i)

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        HEADER_LEN = 64 + 2*length_x*length_y
        # try:
        with gzip.open(target_hsr, 'rb') as f:
            # dBZ start after header.
            f.seek(HEADER_LEN)
            data = f.read()
            raw = data[0:2*length_x*length_y]
            if (len(raw) != 13281410):  # 2 * 2881 * 2305
                raise Exception(f'wrong datasize: {target_hsr}')

            cum_pre = np.frombuffer(raw, 'i2').reshape(length_y, length_x)
            cum_pre = torch.tensor(cum_pre, dtype=torch.float)
            if self.pooling:
                cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), self.pooling, mode='bilinear', align_corners=False).squeeze()

            # if self.normalize_hsr:
            # we cannot unnormalize the predicted output
            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        return cum_pre


class HSRTrain(HSRBase):
    def __init__(self, train_list, wrong_idx_txt, pooling, spatial_to_channel):
        super().__init__(data_list=train_list, wrong_idx_txt=wrong_idx_txt, pooling=pooling, spatial_to_channel=spatial_to_channel)

class HSRTest(HSRBase):
    def __init__(self, test_list, wrong_idx_txt, pooling, spatial_to_channel):
        super().__init__(data_list=test_list, wrong_idx_txt=wrong_idx_txt, pooling=pooling, spatial_to_channel=spatial_to_channel)

class HSRDebug(Dataset):
    def __init__(self, data_list:str, pooling=None, spatial_to_channel=None):
        super().__init__()
        self.data = []
        self.spatial_to_channel = spatial_to_channel
        # hsr params
        self.pooling = False if not pooling else tuple(pooling)

        with open(data_list, "r") as f:
            paths = f.read().splitlines()
        paths = list(sorted(paths))
        print(f"# of Raw Data: {len(paths)}")

        # find wrong idx
        for f_name in tqdm(paths, desc='Data Preparing'):
            self.data.append(f_name)
        print(f"# of Prepared Data: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Simple fix for dirty dataset
        datestr = self.data[i].split('_')[-1].replace('.bin.gz', '')
        assert len(datestr) == 12
        try:
            target = self.parse_hsr(self.data[i])
            if self.spatial_to_channel is not None:
                target = s2c(target, self.spatial_to_channel)
            check=True
        except:
            # hard coded shape
            target = torch.zeros(self.pooling).unsqueeze(-1)
            if self.spatial_to_channel is not None:
                target = s2c(target, self.spatial_to_channel)
            check=False
        return {"data": target, "return": check, "idx": i, "datestr":datestr, }

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        HEADER_LEN = 64 + 2*length_x*length_y
        # try:
        with gzip.open(target_hsr, 'rb') as f:
            # dBZ start after header.
            f.seek(HEADER_LEN)
            data = f.read()
            raw = data[0:2*length_x*length_y]
            if (len(raw) != 13281410):  # 2 * 2881 * 2305
                raise Exception(f'wrong datasize: {target_hsr}')

            cum_pre = np.frombuffer(raw, 'i2').reshape(length_y, length_x)
            cum_pre = torch.tensor(cum_pre, dtype=torch.float)
            if self.pooling:
                cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), self.pooling, mode='bilinear', align_corners=False).squeeze()

            # if self.normalize_hsr:
            # we cannot unnormalize the predicted output
            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        return cum_pre
