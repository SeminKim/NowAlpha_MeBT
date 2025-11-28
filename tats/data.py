# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import os.path as osp
import math
import random
import pickle
import warnings
import re
from datetime import datetime, timedelta

import glob
import h5py
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from taming.data.utils import spatial_to_channel as s2c
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl

# from transformers import BertTokenizer

### For Satellite Dataset
import gzip
import torch.nn.functional as F
from einops import repeat, rearrange
INT_MAX = 2**31

INVALID = -1e-3

def zr_relation(x):
    x = 10**(x*0.1)
    y = (x/148.)**(100./159.)
    return y

def inverse_zr_relation(y):
    y = 148. * (y**(159./100.))
    return y

def normalize_dBZ(x):
    dBz_max = 60.
    dBz_min = 0.
    data = torch.clamp(2.*(x-dBz_min)/(dBz_max-dBz_min)-1.,max = 1)
    return data
#    return torch.tanh(torch.log(x+0.01)/4.)

def unnormalize_dBZ(x):
    dBz_max = 60.
    dBz_min = 0.
    return ((x+1.)*(dBz_max-dBz_min)/2.).clamp(dBz_min, dBz_max)
    # return value: -0.01 ~ 4M
#    return torch.exp(4*torch.arctanh(x.clamp(-0.999, 0.999)))-0.01

def preprocess_precep(x):
    mask = x < 0.
    x = torch.where(mask, INVALID * torch.ones_like(x), x)
    return x
class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64, sample_every_n_frames=1, latent_shape=[]):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames
        self.latent_shape = latent_shape

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)

        # self._clips = clips.subset(np.arange(24))
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        while True:
            try:
                video, _, _, idx = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(**preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames), label=label, indices=torch.randperm(np.prod(self.latent_shape)))


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

def preprocess(video, resolution, sequence_length=None, in_channels=3, sample_every_n_frames=1):
    # video: THWC, {0, ..., 255}
    if in_channels == 3:
        video = video.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # TCHW  # 2DVQGAN!
    else:
        # make the semantic map one hot
        if video.shape[-1] == 3:
            video = video[:, :, :, 0]
        video = F.one_hot(video.long(), num_classes=in_channels).permute(0, 3, 1, 2).float()
        # flatseg = video.reshape(-1)
        # onehot = torch.zeros((flatseg.shape[0], in_channels))
        # onehot[torch.arange(flatseg.shape[0]), flatseg] = 1
        # onehot = onehot.reshape(video.shape + (in_channels,))
        # video = onehot.permute(0, 3, 1, 2).float()
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # skip frames
    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    if in_channels == 3:
        return {'video': video}
    else:
        return {'video_smap': video}

class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(self.args, 'val_vtoken'):
            self.args.val_vtoken = False
        if not hasattr(self.args, 'threshold'):
            self.args.threshold = [0.,0.]
            self.args.second_chance = 1.0

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        if not hasattr(self, 'vtoken_trainset'):
            self.vtoken_trainset = self._dataset(True, True, self.args.threshold, self.args.second_chance)
        if not hasattr(self, 'trainset'):
            self.trainset = self._dataset(True, False, self.args.threshold, self.args.second_chance)
        if not hasattr(self, 'valset'):
            self.valset = self._dataset(False, False)

    def _dataset(self, train, vtoken, threshold=[0.,0.], second_chance=1.0):
        if train:
            if self.args.vqgan:
                spec_args = self.args.train
            else:
                spec_args = self.args.vtoken_train if vtoken else self.args.train
        else:
            spec_args = self.args.test

        if self.args.vqgan:
            dataset = SatelliteDataset4VQGAN (
                hsr_data_path=self.args.hsr_data_path,
                gz_data_path=self.args.gz_data_path,
                include=spec_args.include,
                exclude=spec_args.exclude,
                data_interval=self.args.data_interval,
                input_interval=self.args.input_interval,
                before=self.args.before,
                output_interval=self.args.output_interval,
                after=self.args.after,
                train=train,
                spatial_to_channel=self.args.spatial_to_channel,
                pooling=self.args.pooling,
            )

        elif self.args.COMMIT:
            Dataset = HDF5SatelliteDataset_vtoken_COMMIT if train else HDF5SatelliteDataset_vtoken_COMMIT_eval
            dataset = Dataset(hsr_data_path=spec_args.hsr_data_path,
                                include=spec_args.include,
                                exclude=spec_args.exclude,
                                data_interval=self.args.data_interval,
                                input_interval=self.args.input_interval,
                                before=self.args.before,
                                output_interval=self.args.output_interval,
                                after=self.args.after,
                                use_time=self.args.use_time,
                                latent_shape=self.args.latent_shape,
                                train=train,
                                use_local_stat=hasattr(self.args, 'use_local_stat') and self.args.use_local_stat,
                                threshold=threshold,
                                second_chance=second_chance, 
                                limit_ratio = None if not hasattr(spec_args, 'limit_ratio') else spec_args.limit_ratio,
                                flip=getattr(spec_args, 'flip', ['none']),
                                hsr_interior_path=spec_args.hsr_interior_path,
                                gz_data_path=getattr(spec_args, 'gz_data_path', ['none']),
                                load_dBZ=self.args.load_dBZ
                                )
            
        else:
            assert 0
        return dataset

    def _dataloader(self, train, vtoken=False, shuffle=True):
        if vtoken:
            dataset = self.vtoken_trainset
        else:
            dataset = self.trainset if train else self.valset
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle
            )
        else:
            if hasattr(self.args, 'balanced_sampler') and self.args.balanced_sampler and train:
                sampler = BalancedRandomSampler(dataset.classes_for_sampling)
            else:
                sampler = None
            print(f'Be careful on sampler... now DDP is off and the sampler is {sampler}')
        batch_size = self.args.batch_size
        try:
            if not train and not self.args.vqgan and self.args.val_batch_size:
                batch_size = self.args.val_batch_size
        except:
            pass
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle if sampler is None else None,
            persistent_workers= self.args.num_workers > 0,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True, True, shuffle=True)

    def val_dataloader(self):
        ret = [self._dataloader(False, shuffle=False)]
        return ret

    def test_dataloader(self):
        return self.val_dataloader()


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, default='/datasets01/Kinetics400_Frames/videos')
        parser.add_argument('--sequence_length', type=int, default=16)
        parser.add_argument('--resolution', type=int, default=128)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--image_channels', type=int, default=3)
        parser.add_argument('--smap_cond', type=int, default=0)
        parser.add_argument('--smap_only', action='store_true')
        parser.add_argument('--text_cond', action='store_true')
        parser.add_argument('--vtokens', action='store_true')
        parser.add_argument('--vtokens_pos', action='store_true')
        parser.add_argument('--spatial_length', type=int, default=15)
        parser.add_argument('--sample_every_n_frames', type=int, default=1)
        parser.add_argument('--image_folder', action='store_true')
        parser.add_argument('--stft_data', action='store_true')
        parser.add_argument('--preprocessed_hdf5', action='store_true')

        return parser

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def preprocess_image(image):
    # In 2d VQGAN, range is -1~1
    # In 3d VQGAN (TATS), range is -0.5~0.5
    # Here I am using 2d VQGAN
    # [0, 1] => [-1, 1]
    img = image * 2 - 1.0 # 2DVQGAN!
    img = torch.from_numpy(img)
    return img

class SatelliteDataset(data.Dataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    NOTE: Here we always do shuffle among dataset.
    '''
    def __init__(self, base_dir='/data1/common_datasets/ai_rainfall/CUMUL_DATA', target_date=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], pooling=[1024, 1024], spatial_to_channel=[4, 4], causal=True):
        '''
        threshold: an image should include:
        more than (threshold[0]) 1mm/10mm pixels
        and more than (threshold[1]) 10mm pixels.
        '''
        assert input_interval % data_interval == 0
        self.base_dir = base_dir
        self.latent_shape = latent_shape
        self.data_interval = data_interval
        self.input_interval = input_interval
        self.output_interval = output_interval
        self.before = before
        self.after = after
        self.causal = causal
        self.input_dim = self.before // self.input_interval # ((before // self.data_interval) -1)*(self.input_interval // self.data_interval) + 1
        self.data = [self.get_target_seq(d) for d in target_date]
        self.pooling = pooling
        self.spatial_to_channel = spatial_to_channel

    def get_target_seq(self, target_date):
        date_format_str = "%Y%m%d%H%M"
        datestr = target_date
        read_time = datetime.strptime(datestr, date_format_str)
        # input_times
        input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]

        if self.causal:
            # no future data for inference
            output_times = [read_time + timedelta(minutes=0) for n in range(0, self.after+1, self.output_interval)]
        else:
            output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]
        time_sequence = input_times[1:] + output_times # input + read_time + output
        time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
        time_sequence = [os.path.join(f'{self.base_dir}', t[:6], t[6:8], f'RDR_PCP_HSR_KMA_M060_{t}.bin.gz') for t in time_sequence]
        return time_sequence

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        if self.load_dBZ:
            HEADER_LEN = 1024
        else:
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
            cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), self.pooling, mode='bilinear', align_corners=False).squeeze()
            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        target = cum_pre
        if self.spatial_to_channel is not None:
            target = s2c(target, self.spatial_to_channel)
        mask = target < 0.
        target = torch.where(mask, INVALID * torch.ones_like(target), target)
        return target

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, raw_idx):
        seq = self.data[raw_idx] # Assume 
        # parse_hsr
        radar_history = [self.parse_hsr(s) for s in seq]

        # check the shape
        radar_history = rearrange(torch.stack(radar_history), 't h w c -> c t h w')

        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}
        
        return ret

class HDF5SatelliteDataset(data.Dataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    NOTE: Here we always do shuffle among dataset.
    '''
    def __init__(self, hsr_data_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], train=True, use_time=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, flip=['none']):
        '''
        threshold: an image should include:
        more than (threshold[0]) 1mm/10mm pixels
        and more than (threshold[1]) 10mm pixels.
        '''
        self.flip = flip
        assert input_interval % data_interval == 0
        self.hsr_data_path = hsr_data_path
        self.data = h5py.File(self.hsr_data_path, 'r')
        # TODO: check data shape.
        # if data shape is (B T H W), do not create sequence data

        self.is_vid = len(self.data['data'].shape) == 4

        self.datestr = self.data['datestr']
        self.latent_shape = latent_shape
        self.train = train
        self.data_interval = data_interval
        self.input_interval = input_interval
        self.output_interval = output_interval
        self.include = [re.compile(pat) for pat in include]
        self.exclude = [re.compile(pat) for pat in exclude]
        self.before = before
        self.after = after
        self.input_dim = self.before // self.input_interval # ((before // self.data_interval) -1)*(self.input_interval // self.data_interval) + 1
        self.use_time = use_time
        self.use_local_stat = use_local_stat

        ############################################
        # create datestr2idx & check
        self.datestr2idx = {v.decode('utf-8'): i for i, v in enumerate(self.datestr)}
        self.idx2datestr = {i: v.decode('utf-8') for i, v in enumerate(self.datestr)}
        checklist = list(self.data['check']) # register to the memory to modify
        print(f"NUM of total timestamps: {len(checklist)}")
        ############################################
        
        ############################################
        # Regex check
        for i in range(len(checklist)):
            datestr = self.datestr[i].decode('utf-8')
            include_test = (len(self.include)==0) or any([reg.fullmatch(datestr) is not None for reg in self.include])
            exclude_test = all([reg.fullmatch(datestr) is None for reg in self.exclude])
            regex_test = include_test and exclude_test
            checklist[i] = checklist[i] and regex_test        
        ############################################

        ############################################
        # Create sequence_data
        # This code is for 3d VQ, so we don't need to check the consequent datestrs.
        if not self.is_vid:
            date_format_str = "%Y%m%d%H%M"
            self.sequence_data = []
            self.availables = []
            for datestr in self.datestr:
                ############################################
                # constructe the list of datestrings
                datestr = datestr.decode('utf-8')
                read_time = datetime.strptime(datestr, date_format_str)
                # input_times
                input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]
                # output_times + current_time
                output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]    
                time_sequence = input_times[1:] + output_times
                time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
                ############################################
                
                ############################################
                # check the validity
                try:
                    indices = [self.datestr2idx[timestep] for timestep in time_sequence]
                    valid = all([checklist[idx] for idx in indices])
                    if valid:
                        self.sequence_data.append(indices)
                        self.availables.append(datestr)
                except KeyError:
                    continue
        else:
            self.sequence_data = []
            self.availables = []
            for i, datestr in enumerate(self.datestr):
                if checklist[i]:
                    self.sequence_data.append(i)
                    self.availables.append(datestr.decode('utf-8'))
        # Shuffle
        n = len(self.sequence_data)
        shuffle_idx = np.arange(n)
        np.random.shuffle(shuffle_idx)
        self.sequence_data = np.array(self.sequence_data)[shuffle_idx]
        self.availables = np.array(self.availables)[shuffle_idx]
        self.availables2idx = {str(v): i for i, v in enumerate(self.availables)}

        print(f"NUM of available sequences: {n}, USING: {len(self.sequence_data)}")

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['datestr'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.hsr_data_path, 'r')
        self.datestr = self.data[f'datestr']

    def __len__(self) -> int:
        return len(self.sequence_data)
    
    def parse_dataset(self, target_dates):
        target_indices = [self.availables2idx[date] for date in target_dates if date in self.availables2idx]
        self.sequence_data = self.sequence_data[target_indices]
        return

    def __getitem__(self, raw_idx):
        seq = self.sequence_data[raw_idx]
        # randomly choice flip strategy and apply
        radar_history = rearrange(self.data['data'][seq], 't h w c -> c t h w')

        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}
        
        # After collate, the shape is
        # video: (B) T H W <-> (?) 13 16 16
        # input_indcies: (B) N <-> (?) 1792
        # input_indcies: (B) N <-> (?) 1536
        # mask: (B) T H W <-> (?) 13 16 16
        # time: T (B) <-> 13 (?)  // NOTE: this is list of list of string

        return ret

class HDF5SatelliteDataset_vtoken(HDF5SatelliteDataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    '''
    def __getitem__(self, raw_idx):
        seq = self.sequence_data[raw_idx]
        # T H W
        # randomly choice flip strategy and apply
        radar_history = self.data['data'][seq]

        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}
        
        # After collate, the shape is
        # video: (B) T H W <-> (?) 13 16 16
        # input_indcies: (B) N <-> (?) 1792
        # input_indcies: (B) N <-> (?) 1536
        # mask: (B) T H W <-> (?) 13 16 16
        # time: T (B) <-> 13 (?)  // NOTE: this is list of list of string

        return ret


class HDF5SatelliteDataset_vtoken_eval(HDF5SatelliteDataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    '''
    def __init__(self, hsr_data_path=None, gz_data_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360, use_time=True,
                 latent_shape=[], train=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, eval_t=[60, 120, 180], load_dBZ=False):
        '''
        threshold: an image should include:
        more than (threshold[0]) 1mm/10mm pixels
        and more than (threshold[1]) 10mm pixels.
        '''
        super().__init__(hsr_data_path, include, exclude, data_interval, input_interval, before, output_interval,
                         after, latent_shape, train, use_time, use_local_stat, threshold, second_chance, limit_ratio)

        self.str_format = "{data_path}/{yyyymm}/{dd}/RDR_PCP_HSR_KMA_M060_{yyyymm}{dd}{hourmin}.bin.gz"
        self.str_format_KMA = "{data_path}/{yyyymm}/{dd}/RDR_CMP_HSR_KMA_{yyyymm}{dd}{hourmin}.bin.gz"
        self.str_format_CPP = "{data_path}/{yyyymm}/RDR_CMP_CPP_QCD_{yyyymm}{dd}{hourmin}.bin.gz"
        self.gz_data_path = gz_data_path
        self.eval_t = [t for t in range(-before, after, output_interval)]
        self.eval_idx = list(range(len(self.eval_t)))
        self.load_dBZ = load_dBZ

    def __getitem__(self, raw_idx):
        seq = self.sequence_data[raw_idx]
        # T H W
        radar_history = self.data['data'][seq]
        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}

        # load target_file
        gts = []
        date_format_str = "%Y%m%d%H%M"
        datestr = self.datestr[seq].decode('utf-8')
        read_time = datetime.strptime(datestr, date_format_str)
        # input_times
        input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]
        # output_times + current_time
        output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]    
        time_sequence = input_times[1:] + output_times
        time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
        for i in self.eval_idx:
            datestr = time_sequence[i]
            yyyymm = datestr[:6]
            yyyy = datestr[:4]
            if self.load_dBZ:
                if yyyy in ['2014', '2015', '2016', '2017']:
                    str_format = self.str_format_CPP
                else:
                    str_format = self.str_format_KMA
            else:
                str_format = self.str_format
            dd = datestr[6:8]
            hourmin = datestr[8:]
            fname = str_format.format(data_path=self.gz_data_path, yyyymm=yyyymm, dd=dd, hourmin=hourmin)
            gt = self.parse_hsr(fname)
            gts.append(gt)

        ret['gts'] = torch.stack(gts)
        ret['time'] = time_sequence
        ret['eval_idx'] = torch.tensor(self.eval_idx).long()
        return ret

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        if self.load_dBZ:
            HEADER_LEN = 1024
        else:
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
            cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), scale_factor=0.5, recompute_scale_factor=True).squeeze()

            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        return cum_pre

class SatelliteDataset4VQGAN(data.Dataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    '''
    def __init__(self, hsr_data_path=None, gz_data_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360, use_time=True,
                 latent_shape=[],train=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, eval_t=[60, 120, 180],
                 spatial_to_channel=(2,2), pooling=(1024, 1024)):
        '''
        threshold: an image should include:
        more than (threshold[0]) 1mm/10mm pixels
        and more than (threshold[1]) 10mm pixels.
        '''

        self.str_format_KMA = "{data_path}/{yyyymm}/{dd}/RDR_CMP_HSR_KMA_{yyyymm}{dd}{hourmin}.bin.gz"
        self.str_format_CPP = "{data_path}/{yyyymm}/RDR_CMP_CPP_QCD_{yyyymm}{dd}{hourmin}.bin.gz"
        self.gz_data_path = gz_data_path
        assert input_interval % data_interval == 0
        self.hsr_data_path = hsr_data_path
        self.data = h5py.File(self.hsr_data_path, 'r')
        self.datestr = self.data['datestr']
        self.latent_shape = latent_shape
        self.train = train
        self.data_interval = data_interval
        self.input_interval = input_interval
        self.output_interval = output_interval
        self.include = [re.compile(pat) for pat in include]
        self.exclude = [re.compile(pat) for pat in exclude]
        self.before = before
        self.after = after
        self.input_dim = self.before // self.input_interval # ((before // self.data_interval) -1)*(self.input_interval // self.data_interval) + 1
        self.use_time = use_time
        self.spatial_to_channel = spatial_to_channel
        self.pooling = tuple(pooling)
        
        print(f"NUM of total timestamps: {len(self.data['check'])}")

        ############################################
        # load cache if cached
        non_rgx_include = [x.replace('.', '').replace('*', '') for x in include]
        non_rgx_exclude = [x.replace('.', '').replace('*', '') for x in exclude]
        cache_path = self.hsr_data_path + '.cache' + f'.{before}.{after}.{input_interval}.{output_interval}.' + '.'.join(non_rgx_include) + '.' + '.'.join(non_rgx_exclude)
        load_cache = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    _include, _exclude, self.sequence_data, self.availables, self.datestr2idx = pickle.load(f)
                    load_cache = True
                    print("Cache loaded from", cache_path)
            except:
                load_cache = False
        if not load_cache:
            print("Cache not found. Creating cache...")
            ############################################
            # create datestr2idx & check
            self.datestr2idx = {v.decode('utf-8'): i for i, v in enumerate(self.datestr)}
            checklist = list(self.data['check']) # register to the memory to modify
            ############################################
            ############################################
            # Regex check
            for i in range(len(checklist)):
                datestr = self.datestr[i].decode('utf-8')
                include_test = (len(self.include)==0) or any([reg.fullmatch(datestr) is not None for reg in self.include])
                exclude_test = all([reg.fullmatch(datestr) is None for reg in self.exclude])
                regex_test = include_test and exclude_test
                checklist[i] = checklist[i] and regex_test        
            ############################################
            # Create sequence_data
            date_format_str = "%Y%m%d%H%M"
            self.sequence_data = []
            self.availables = []
            for datestr in self.datestr:
                ############################################
                # construct the list of datestrings
                datestr = datestr.decode('utf-8')
                read_time = datetime.strptime(datestr, date_format_str)
                # input_times
                input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]
                # output_times + current_time
                output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]    
                time_sequence = input_times[1:] + output_times
                time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
                ############################################
                
                ############################################
                # check the validity
                try:
                    indices = [self.datestr2idx[timestep] for timestep in time_sequence]
                    valid = all([checklist[idx] for idx in indices])
                    if valid:
                        self.sequence_data.append(indices)
                        self.availables.append(datestr)
                except KeyError:
                    continue
            
            # save cache
            with open(cache_path, 'wb') as f:
                pickle.dump((include, exclude, self.sequence_data, self.availables, self.datestr2idx), f)
            print("Cache saved to", cache_path)

        # Shuffle
        n = len(self.sequence_data)
        shuffle_idx = np.arange(n)
        np.random.shuffle(shuffle_idx)
        self.sequence_data = np.array(self.sequence_data)[shuffle_idx]
        self.availables = np.array(self.availables)[shuffle_idx]
        self.availables2idx = {v: i for i, v in enumerate(self.availables)}

        print(f"NUM of available sequences: {n}, USING: {len(self.sequence_data)}")
    
    def parse_dataset(self, target_dates, ignore=False):
        if ignore:
            target_indices = [self.availables2idx[date] for date in target_dates if date not in self.availables2idx]
        else:
            target_indices = [self.availables2idx[date] for date in target_dates if date in self.availables2idx]
        self.sequence_data = self.sequence_data[target_indices]
        print(f"USING: {len(self.sequence_data)}")
        return

    def __getitem__(self, raw_idx):
        # TODO: we don't need VQ data. use hdf5 only for datestr
        seq = self.sequence_data[raw_idx]
        # T H W
        # radar_history = self.data['data'][seq]
        # frame_size = np.prod(self.latent_shape[1:])
        # pred_len = self.latent_shape[0] - self.input_dim
        # input_indices = torch.arange(frame_size * self.input_dim)
        # pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        # ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}

        # load target_file (We only need this!)
        gts = []
        masks = []
        for idx in seq:
            datestr = self.datestr[idx].decode('utf-8')
            yyyy = datestr[:4]
            yyyymm = datestr[:6]
            dd = datestr[6:8]
            hourmin = datestr[8:]
            if yyyy in ['2014', '2015', '2016', '2017']:
                str_format = self.str_format_CPP
            else:
                str_format = self.str_format_KMA
            fname = str_format.format(data_path=self.gz_data_path, yyyymm=yyyymm, dd=dd, hourmin=hourmin)
            gt = self.parse_hsr(fname)
            if self.spatial_to_channel is not None:
                gt = s2c(gt, self.spatial_to_channel)
            mask = gt < 0.
            masks.append(mask)
            gt = torch.where(mask, INVALID * torch.ones_like(gt), gt)
            gts.append(gt)

        ret = {}
        ret['video'] = rearrange(torch.stack(gts), 't h w c -> c t h w')
        if torch.isnan(ret['video']).any():
            assert 0
        ret['mask'] = rearrange(torch.stack(masks), 't h w c -> c t h w')
        # ret['eval_idx'] = torch.tensor(self.eval_idx).long()
        ret['time'] = [self.datestr[i].decode('utf-8') for i in seq]
        return ret

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        HEADER_LEN = 1024
        # try:
        with gzip.open(target_hsr, 'rb') as f:
            # dBZ start after header.
            try:
                f.seek(HEADER_LEN)
            except:
                print(target_hsr)
                assert 0
            data = f.read()
            raw = data[0:2*length_x*length_y]
            if (len(raw) != 13281410):  # 2 * 2881 * 2305
                raise Exception(f'wrong datasize: {target_hsr}')

            cum_pre = np.frombuffer(raw, 'i2').reshape(length_y, length_x)
            cum_pre = torch.tensor(cum_pre, dtype=torch.float)
            cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), self.pooling, mode='bilinear', align_corners=False).squeeze()
            # cum_pre = resize(cum_pre.unsqueeze(0).unsqueeze(0), size=self.pooling, interpolation=InterpolationMode.BILINEAR).squeeze()

            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        return cum_pre

    def __len__(self):
        return len(self.sequence_data)

class SatelliteDataset4VQGAN_infer(data.Dataset):
    '''
    Dataset for HSR files (Including PUB)
    For original code, see https://github.com/deukryeol-yoon/weather-satellite/blob/MetNet_master/precipitation/dataset.py
    '''
    def __init__(self, gz_data_path=None, target_dates=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], spatial_to_channel=(4,4), pooling=(1024, 1024), load_future=False):
        '''
        threshold: an image should include:
        more than (threshold[0]) 1mm/10mm pixels
        and more than (threshold[1]) 10mm pixels.
        '''

        self.str_format_KMA = "{data_path}/{yyyymm}/{dd}/RDR_CMP_HSR_KMA_{yyyymm}{dd}{hourmin}.bin.gz"
        self.str_format_CPP = "{data_path}/{yyyymm}/RDR_CMP_CPP_QCD_{yyyymm}{dd}{hourmin}.bin.gz"
        self.gz_data_path = gz_data_path
        assert input_interval % data_interval == 0
        self.target_dates = target_dates
        self.latent_shape = latent_shape
        self.data_interval = data_interval
        self.input_interval = input_interval
        self.output_interval = output_interval
        self.before = before
        self.after = after
        self.input_dim = self.before // self.input_interval # ((before // self.data_interval) -1)*(self.input_interval // self.data_interval) + 1
        self.spatial_to_channel = spatial_to_channel
        self.pooling = tuple(pooling)
        self.load_future = load_future
        
        ############################################
        # create datestr2idx & check
        ############################################
        # Create sequence_data
        date_format_str = "%Y%m%d%H%M"
        self.sequence_data = []
        self.availables = []
        for datestr in self.target_dates:
            ############################################
            # construct the list of datestrings
            read_time = datetime.strptime(datestr, date_format_str)
            # input_times
            input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]
            # output_times + current_time
            if not self.load_future:
                output_times = [read_time + timedelta(minutes=0) for n in range(0, self.after+1, self.output_interval)]
            else:
                output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]
            time_sequence = input_times[1:] + output_times
            time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
            ############################################
            
            ############################################
            # check the validity
            valid = all([os.path.exists(self.str_format_CPP.format(data_path=self.gz_data_path, yyyymm=datestr[:6], dd=datestr[6:8], hourmin=datestr[8:]))
                        or os.path.exists(self.str_format_KMA.format(data_path=self.gz_data_path, yyyymm=datestr[:6], dd=datestr[6:8], hourmin=datestr[8:]))
                        for datestr in time_sequence])
            if valid:
                self.sequence_data.append(time_sequence)
                self.availables.append(datestr)
            else:
                print(f"Invalid: {datestr}")
                print(self.str_format_KMA.format(data_path=self.gz_data_path, yyyymm=datestr[:6], dd=datestr[6:8], hourmin=datestr[8:]))
        print(f"NUM of available sequences: {len(self.sequence_data)}")
    
    def __getitem__(self, raw_idx):
        # T H W
        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.arange(frame_size * pred_len) + frame_size * self.input_dim
        seq = self.sequence_data[raw_idx]
        # load target_file (We only need this!)
        gts = []
        masks = []
        for idx in seq:
            datestr = idx
            yyyy = datestr[:4]
            yyyymm = datestr[:6]
            dd = datestr[6:8]
            hourmin = datestr[8:]
            if yyyy in ['2014', '2015', '2016', '2017']:
                str_format = self.str_format_CPP
            else:
                str_format = self.str_format_KMA
            fname = str_format.format(data_path=self.gz_data_path, yyyymm=yyyymm, dd=dd, hourmin=hourmin)
            try:
                gt = self.parse_hsr(fname)
            except Exception as e:
                print(e)
                return {'check': False, 'fname':fname}
            masks.append(gt < -250)
            if self.spatial_to_channel is not None:
                gt = s2c(gt, self.spatial_to_channel)
            mask = gt < 0.
            gt = torch.where(mask, INVALID * torch.ones_like(gt), gt)
            gts.append(gt)

        ret = {}
        ret['video'] = rearrange(torch.stack(gts), 't h w c -> c t h w')
        if torch.isnan(ret['video']).any():
            assert 0
        ret['mask'] = rearrange(torch.stack(masks), 't h w c -> c t h w')
        ret['time'] = seq
        ret['input_indices'] = input_indices
        ret['pred_indices'] = pred_indices
        eval_idx = list(range(self.latent_shape[0]))
        ret['eval_idx'] = torch.tensor(eval_idx).long()
        context_len = self.before // self.input_interval
        ret['datestr'] = seq[context_len-1]
        ret['check'] = True
        return ret

    def parse_hsr(self, target_hsr):
        length_x = 2305
        length_y = 2881
        HEADER_LEN = 1024
        # try:
        with gzip.open(target_hsr, 'rb') as f:
            # dBZ start after header.
            try:
                f.seek(HEADER_LEN)
            except:
                print(target_hsr)
                assert 0
            data = f.read()
            raw = data[0:2*length_x*length_y]
            if (len(raw) != 13281410):  # 2 * 2881 * 2305
                raise Exception(f'wrong datasize: {target_hsr}')

            cum_pre = np.frombuffer(raw, 'i2').reshape(length_y, length_x)
            cum_pre = torch.tensor(cum_pre, dtype=torch.float)
            cum_pre = F.interpolate(cum_pre.unsqueeze(0).unsqueeze(0), self.pooling, mode='bilinear', align_corners=False).squeeze()
            # cum_pre = resize(cum_pre.unsqueeze(0).unsqueeze(0), size=self.pooling, interpolation=InterpolationMode.BILINEAR).squeeze()

            cum_pre = cum_pre / 100.
            cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        return cum_pre

    def __len__(self):
        return len(self.sequence_data)

class HDF5SatelliteDataset_vtoken_vis(HDF5SatelliteDataset_vtoken_eval):
    def __getitem__(self, raw_idx):
        eval_idx = list(range(self.latent_shape[0]))
        seq = self.sequence_data[raw_idx]
        # T H W
        radar_history = self.data['data'][seq]
        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.randperm(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history, 'input_indices': input_indices.long(), 'pred_indices': pred_indices.long(), 'mask': radar_history < 0.}

        # load target_file
        gts = []
        raws = []

        date_format_str = "%Y%m%d%H%M"
        datestr = self.datestr[seq]
        if isinstance(datestr, np.ndarray):
            datestr = datestr[6].decode('utf-8')
        else:
            datestr = datestr.decode('utf-8')
        read_time = datetime.strptime(datestr, date_format_str)
        # input_times
        input_times = [read_time + timedelta(minutes=n) for n in range(-self.before, 0, self.input_interval)]
        # output_times + current_time
        output_times = [read_time + timedelta(minutes=n) for n in range(0, self.after+1, self.output_interval)]    
        time_sequence = input_times[1:] + output_times
        time_sequence = [timestr.strftime(date_format_str) for timestr in time_sequence]
        for i in eval_idx:
            datestr = time_sequence[i]
            yyyymm = datestr[:6]
            yyyy = datestr[:4]
            if self.load_dBZ:
                if yyyy in ['2014', '2015', '2016', '2017']:
                    str_format = self.str_format_CPP
                else:
                    str_format = self.str_format_KMA
            else:
                str_format = self.str_format
            dd = datestr[6:8]
            hourmin = datestr[8:]
            fname = str_format.format(data_path=self.gz_data_path, yyyymm=yyyymm, dd=dd, hourmin=hourmin)
            gt = self.parse_hsr(fname)
            gts.append(gt)

        ret['gts'] = torch.stack(gts)
        ret['time'] = time_sequence
        ret['eval_idx'] = torch.tensor(eval_idx).long()
        return ret

class HDF5SatelliteDataset_vtoken_COMMIT(HDF5SatelliteDataset):
    '''
    Dataset for HSR files (Including PUB)
    Load vq tokens from both raw video and interior video
    '''

    def __init__(self, hsr_data_path=None, hsr_interior_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], train=True, use_time=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, flip=['none'], load_dBZ=False, *args, **kwargs):
        super().__init__(hsr_data_path, include, exclude, data_interval, input_interval, before, output_interval, after,
                 latent_shape, train, use_time, use_local_stat, threshold, second_chance, limit_ratio, flip)
        self.hsr_interior_path = hsr_interior_path
        self.interior_data = h5py.File(self.hsr_interior_path, 'r')
        self.interior_datestr = self.interior_data['datestr']
        self.vid_length = (before+after) // input_interval
        self.interior_datestr2idx = {v.decode('utf-8'): i for i, v in enumerate(self.interior_datestr)}

    def __getitem__(self, raw_idx):
        seq = self.sequence_data[raw_idx]
        datestr = self.idx2datestr[seq]
        interior_seq = self.interior_datestr2idx[datestr]

        radar_history = self.data['data'][seq] # T H W
        interior_data = self.interior_data['data'][interior_seq] # T H W

        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        input_indices = torch.arange(frame_size * self.input_dim)
        pred_indices = torch.arange(frame_size * pred_len) + frame_size * self.input_dim
        ret = {'video': radar_history[:self.vid_length], 
               'interior_video': interior_data[:self.vid_length],
               'input_indices': input_indices.long(), 
               'pred_indices': pred_indices.long(), 
               'mask': radar_history < 0.}

        return ret

class HDF5SatelliteDataset_vtoken_COMMIT_eval(HDF5SatelliteDataset_vtoken_eval):
    '''
    Dataset for HSR files (Including PUB)
    Load vq tokens from both raw video and interior video
    '''

    def __init__(self, hsr_data_path=None, gz_data_path=None, hsr_interior_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], train=True, use_time=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, eval_t=[60, 120, 180], load_dBZ=False, *args, **kwargs):
        super().__init__(hsr_data_path, gz_data_path, include, exclude, data_interval, input_interval, before, output_interval, after,
                         use_time, latent_shape, train, use_local_stat, threshold, second_chance, limit_ratio,
                         eval_t, load_dBZ)
        self.hsr_interior_path = hsr_interior_path
        self.interior_data = h5py.File(self.hsr_interior_path, 'r')
        self.interior_datestr = self.interior_data['datestr']
        self.interior_datestr2idx = {v.decode('utf-8'): i for i, v in enumerate(self.interior_datestr)}
        self.vid_length = (before+after) // input_interval


    def __getitem__(self, raw_idx):
        ret = super().__getitem__(raw_idx)
        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        pred_indices = torch.arange(frame_size * pred_len) + frame_size * self.input_dim
        ret['pred_indices'] = pred_indices.long()
        seq = self.sequence_data[raw_idx]
        datestr = self.idx2datestr[seq]
        interior_seq = self.interior_datestr2idx[datestr]
        interior_data = self.interior_data['data'][interior_seq] # T H W
        ret['interior_video'] = interior_data[:self.vid_length]
        return ret

class HDF5SatelliteDataset_vtoken_COMMIT_vis(HDF5SatelliteDataset_vtoken_vis):
    '''
    Dataset for HSR files (Including PUB)
    Load vq tokens from both raw video and interior video
    '''

    def __init__(self, hsr_data_path=None, gz_data_path=None, hsr_interior_path=None, include=[], exclude=[], data_interval=10,
                 input_interval=10, before=70, output_interval=60, after=360,
                 latent_shape=[], train=True, use_time=True, use_local_stat=False, threshold=[0.,0.], second_chance=1.0, limit_ratio=None, eval_t=[60, 120, 180],
                 load_dBZ=False):
        super().__init__(hsr_data_path, gz_data_path, include, exclude, data_interval, input_interval, before, output_interval, after,
                         use_time, latent_shape, train, use_local_stat, threshold, second_chance, limit_ratio,
                         eval_t, load_dBZ)
        self.hsr_interior_path = hsr_interior_path
        self.interior_data = h5py.File(self.hsr_interior_path, 'r')
        self.interior_datestr = self.interior_data['datestr']
        self.interior_datestr2idx = {v.decode('utf-8'): i for i, v in enumerate(self.interior_datestr)}


    def __getitem__(self, raw_idx):
        ret = super().__getitem__(raw_idx)
        frame_size = np.prod(self.latent_shape[1:])
        pred_len = self.latent_shape[0] - self.input_dim
        pred_indices = torch.arange(frame_size * pred_len) + frame_size * self.input_dim
        ret['pred_indices'] = pred_indices.long()
        seq = self.sequence_data[raw_idx]
        datestr = self.idx2datestr[seq]
        interior_seq = self.interior_datestr2idx[datestr]
        interior_data = self.interior_data['data'][interior_seq] # T H W
        ret['interior_video'] = interior_data
        return ret
