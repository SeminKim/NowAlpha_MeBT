# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch
import os
import gzip
import re
from matplotlib import font_manager
import imageio
from PIL import Image

import math
import numpy as np
import skvideo.io
from typing import List, Tuple
from datetime import datetime, timedelta

import sys
import pdb as pdb_original
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision

# module for visualization
import matplotlib
from mpl_toolkits.basemap import Basemap

color = ['#FAFAFA','#00C8FF','#009BF5','#004AF5','#00FF00','#00BE00','#008C00','#005A00','#FFFF00','#FFDC1F','#F9CD00','#E0B900','#CCAA00','#FF6600','#FF3200','#D20000','#B40000','#E0A9FF','#C969FF','#B329FF','#9300E4','#B3B4DE','#4C4EB1','#000390','#333333'] # new
cla_num = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #new
cls_thres = [0.09,0.1,0.5,1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,90,110,150,200] #new
cmap = matplotlib.colors.ListedColormap(color)

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def visualize_dbz_to_png(input_path, output_path, region='H3', no_boundary=False):
    print(f"Reading OTH file: {input_path}")
    length_x = 3521
    length_y = 3521
    HEADER_LEN = 1024
    crop_size = (1536, 1536) # As specified
    BOUNDARY_DATA_PATH = '/home/work/RADAR/workspace/ineeji/east_asia/hsr_oth_boundary/boundary_coordinates.npz'
    FONT_PATH = '/home/work/RADAR/workspace/ineeji/DRAW_RADAR/ANCIL/NanumBarunGothic.ttf'
    # Load boundary data
    boundary_data = None
    if os.path.exists(BOUNDARY_DATA_PATH):
        boundary_data = np.load(BOUNDARY_DATA_PATH)
    # Global cache for Basemap
    BASEMAP_CACHE = {}
    # Font settings
    if os.path.exists(FONT_PATH):
        font_prop = font_manager.FontProperties(fname=FONT_PATH)
    else:
        font_prop = None
        
    def zr_relation(x):
        x = 10**(x*0.1)
        y = (x/148.)**(100./159.)
        return y
    def create_lcc_basemap_from_offset_east_asia(lat_0, lon_0,
                                            width, height,
                                            x_offset, y_offset,
                                            lat_1=30, lat_2=60,
                                            resolution='l'):
        tmp_map = Basemap(
            width=width, height=height,
            projection='lcc',
            lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            resolution=resolution
        )
        center_x, center_y = tmp_map(lon_0, lat_0)
        ll_x = center_x + x_offset + 1200000
        ll_y = center_y + y_offset + 1400000
        ur_x = ll_x + 1536000
        ur_y = ll_y + 1536000
        ll_lon, ll_lat = tmp_map(ll_x, ll_y, inverse=True)
        ur_lon, ur_lat = tmp_map(ur_x, ur_y, inverse=True)
        final_map = Basemap(
            projection='lcc',
            llcrnrlon=ll_lon, llcrnrlat=ll_lat,
            urcrnrlon=ur_lon, urcrnrlat=ur_lat,
            lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            resolution=resolution
        )
        return final_map
    def get_or_create_basemap(region='H3'):
        if region not in BASEMAP_CACHE:
            print(f"Creating new Basemap for region '{region}'...")
            BASEMAP_CACHE[region] = get_basemap_by_region(region)
            print(f"Basemap for region '{region}' created and cached.")
        return BASEMAP_CACHE[region]
    def get_basemap_by_region(region='H3'):
        region_params = {
            "H3": {
                "resolution": 'i',
                "width": 3520000,
                "height": 3520000,
                "lat_0": 38.0,
                "lon_0": 126.0,
                "x_offset": -2000000,
                "y_offset": -2400000,
            },
        }
        region = region.upper()
        if region not in region_params:
            raise ValueError(f"Unsupported region: {region}")
        p = region_params[region]
        return create_lcc_basemap_from_offset_east_asia(
            lat_0=p["lat_0"],
            lon_0=p["lon_0"],
            width=p["width"],
            height=p["height"],
            x_offset=p["x_offset"],
            y_offset=p["y_offset"],
            lat_1=p["lat_0"],
            lat_2=p["lat_0"],
            resolution=p["resolution"]
        )
    def visualize_frame(data_frame, datestr, output_path, region='H3',
                        frame_time_offset=0, mode='OBS', basemap=None, 
                        base_time_utc=None, show_boundary=True):
        
        # Convert precipitation (mm/hr) to classes
        gt_cls = np.zeros_like(data_frame)
        cls_idx = 1
        for ct in cls_thres:
            gt_cls[data_frame >= ct] = cls_idx
            cls_idx += 1
        
        if basemap is None:
            m = get_or_create_basemap(region)
        else:
            m = basemap
        # import ipdb; ipdb.set_trace()
        length_y, length_x = data_frame.shape
        x = np.linspace(m.llcrnrx, m.urcrnrx, length_x)
        y = np.linspace(m.llcrnry, m.urcrnry, length_y)
        x2d, y2d = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        m.drawcoastlines(linewidth=0.6, color='gray')
        m.drawcountries(linewidth=0.8, color='gray')
        cs = m.contourf(x2d, y2d, gt_cls, levels=cla_num, cmap=cmap)
        if show_boundary and boundary_data is not None and region in ['H3', 'CROP', 'eastasia']:
            oth_boundary_x = boundary_data['oth_boundary_x']
            oth_boundary_y = boundary_data['oth_boundary_y']
            m.plot(oth_boundary_x, oth_boundary_y, 'r-', linewidth=2.0,
                label='HSR Boundary', alpha=0.8, zorder=10)
            plt.legend(loc='upper left', fontsize=10, framealpha=0.7)
        
        cbar = m.colorbar(cs, location='right', pad='5%')
        cbar.set_ticks(cla_num)
        cbar.set_ticklabels(['', '0', '0.1', '0.5', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                            '15', '20', '25', '30', '40', '50', '60', '70', '90', '110', '150', '200'])
        cbar.set_label('Precipitation (mm/hr)', rotation=270, labelpad=20, fontsize=12)
        
        # Time processing
        base_time = datetime.strptime(datestr[:12], "%Y%m%d%H%M")
        frame_time = base_time + timedelta(minutes=frame_time_offset)
        
        obstime_kst = frame_time
        obstime_utc = obstime_kst - timedelta(hours=9)
        obstime_disp = obstime_kst.strftime("%H:%M KST %d %b %Y")
        
        if base_time_utc:
            obstime_utc_disp = base_time_utc
        else:
            obstime_utc_disp = obstime_utc.strftime("%H:%M UTC %d %b %Y")
        
        plt.title(obstime_disp, loc='right', color='red', fontproperties=font_prop, fontsize=17)
        plt.title(obstime_utc_disp, loc='left', y=-0.05, fontproperties=font_prop, fontsize=17)
        
        time_label = "0min"
        plt.annotate(time_label, xy=(0.02, 0.02), xycoords='axes fraction',
                    fontproperties=font_prop, fontsize=15, color='black')
        
        label_color = 'blue'
        plt.text(0.0, 1.015, mode.upper(), transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=label_color,
                va='bottom', ha='left', fontproperties=font_prop,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=label_color, alpha=0.8))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close()
        
        return output_path
    
    try:
        with gzip.open(input_path, 'rb') as f:
            f.seek(HEADER_LEN)
            data = f.read()
        ll_corner=(1400, 1200)
        raw = data[0:2*length_x*length_y]
        if (len(raw) != 24794882):  # 2 * 3521 * 3521
            raise Exception(f'wrong datasize')
        cum_pre = np.frombuffer(raw, 'i2').reshape(length_y, length_x)
        cum_pre = torch.tensor(cum_pre, dtype=torch.float)
        crop_h, crop_w = crop_size
        
        y1 = ll_corner[0]
        y2 = ll_corner[0] + crop_h
        x1 = ll_corner[1]
        x2 = ll_corner[1] + crop_w
        cum_pre = cum_pre[y1:y2, x1:x2,] # crop
        
        # if self.pooling:
        #     cum_pre = resize(cum_pre.unsqueeze(0).unsqueeze(0), size=self.pooling, interpolation=InterpolationMode.BILINEAR).squeeze()
        cum_pre = cum_pre / 100.
        # cum_pre = cum_pre.reshape((*cum_pre.shape, 1)) # single channel
        
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return
        
    # print(f"  File loaded. Original shape -> Resized to {cum_pre.shape}")
    # print("\n" + "="*30)
    # print("dBZ Data Stats (After loading)")
    # print(f"  Min:    {cum_pre.min().item():.2f}")
    # print(f"  Max:    {cum_pre.max().item():.2f}")
    # print(f"  Mean:   {cum_pre.mean().item():.2f}")
    # print("="*30 + "\n")
    # print("Applying Z-R relation (dBZ -> mm/hr)")
    data_mmhr_tensor = zr_relation(cum_pre)
    # print("\n" + "="*30)
    # print("mm/hr Data Stats (After Z-R)")
    # print(f"  Min:    {data_mmhr_tensor.min().item():.4f}")
    # print(f"  Max:    {data_mmhr_tensor.max().item():.4f}")
    # print(f"  Mean:   {data_mmhr_tensor.mean().item():.4f}")
    # print("="*30 + "\n")
    data_numpy = data_mmhr_tensor.numpy()
    basename = os.path.basename(input_path)
    match = re.search(r'(\d{12})', basename)
    datestr = match.group(1) if match else "200001010000"
    # print(f"  Datestr: {datestr}")
    m = get_or_create_basemap(region)
    print(f"Visualizing frame and saving to: {output_path}")
    visualize_frame(
        data_numpy,
        datestr,
        output_path,
        region=region,
        frame_time_offset=0,
        mode='Observation',
        basemap=m,
        show_boundary=not no_boundary
    )
    
    print(f"Visualizing frame and saved to: {output_path}")

# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight

def save_image_grid(video, fname, nrow=None, fps=6):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.imsave(fname, video[0])
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    print('saved videos to', fname)

def figure_to_array(figure):
    figure.canvas.draw()
    array = np.array(figure.canvas.renderer._renderer)[:, :, :3]
    return array

def visualize_zr(zr, vmax=10):
    # zr: b c t h w
    if len(zr.shape) == 5:
        zr = rearrange(zr, 'b c t h w -> t b c h w')
        tmp = []
        for img in zr:
            grid = torchvision.utils.make_grid(img, nrow=img.shape[0], padding=20, pad_value=10)
            grid = grid[0].float().cpu().numpy()

            gt_cls = np.zeros_like(grid)
            for i, ct in enumerate(cls_thres):
                gt_cls[grid > ct] = i+1

            # draw contourf
            # rows: batch, cols: 2
            B = img.shape[0]
            plt.figure(figsize=(5*B, 5))
            plt.contourf(gt_cls, levels=cla_num, cmap=cmap)
            cbar = plt.colorbar()
            cbar.set_ticks(cla_num) #9
            cbar.set_ticklabels(['','0','0.1','0.5','1','2','3','4','5','6','7','8','9','10','15','20','25','30','40','50','60','70','90','110','150','200']) #new
            cbar.ax.tick_params(labelsize=15)
            plt.axis('off')
            plt.tight_layout()
            
            array = figure_to_array(plt.gcf())
            tmp.append(array)
            plt.close()
        return rearrange(np.stack(tmp), 't h w c -> t c h w')
        # return rearrange(np.stack(tmp), '(t b) h w c -> b t c h w', b=1)
    elif len(zr.shape) == 4:
        b, c, h, w = zr.shape
        grid = torchvision.utils.make_grid(zr, nrow=zr.shape[0], padding=20, pad_value=10)
        grid = grid[0].float().cpu().numpy()

        gt_cls = np.zeros_like(grid)
        for i, ct in enumerate(cls_thres):
            gt_cls[grid > ct] = i+1

        # draw contourf
        # rows: batch, cols: 2
        B = img.shape[0]
        plt.figure(figsize=(5*B, 5))
        plt.contourf(gt_cls, levels=cla_num, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_ticks(cla_num) #9
        cbar.set_ticklabels(['','0','0.1','0.5','1','2','3','4','5','6','7','8','9','10','15','20','25','30','40','50','60','70','90','110','150','200']) #new
        cbar.ax.tick_params(labelsize=15)
        plt.axis('off')
        plt.tight_layout()
        
        array = figure_to_array(plt.gcf())
        plt.close()
        return array

def save_video_grid(video, fname, nrow=None, fps=6):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    video = [Image.fromarray(vid, 'RGB') for vid in video]
    video[0].save(fname, quality=95, save_all=True, append_images=video[1:], fps=fps, loop=0)
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default

def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""

def spatial_to_channel(x, comp_ratio=(1,1)):
    '''
    Do spatial compression by flattening spatial patch in to channel.
    example:
    foo = torch.tensor([[[[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11],
                       [12, 13, 14, 15]]]]) # shape: (1,1,4,4)
    spatial_to_channel(foo, (2,2))
    >> tensor([[[[ 0,  2],
                 [ 8, 10]]],

               [[[ 1,  3],
                 [ 9, 11]]],

               [[[ 4,  6],
                 [12, 14]]],

               [[[ 5,  7],
                 [13, 15]]]]) # shape: (4,1,2,2)
    '''
    assert len(x.shape) == 4, x.shape
    assert len(comp_ratio) == 2
    return rearrange(x, pattern='c t (h1 h2) (w1 w2) -> (c h2 w2) t h1 w1', h2=comp_ratio[0], w2=comp_ratio[1])

@torch.no_grad()
def f1_csi_from_conf_mat(conf_mat: torch.Tensor, aggregate=False) -> dict:
    if len(conf_mat.shape) == 4:
        conf_mat = conf_mat.sum(0)

    # TODO: generalize to T, N, N
    # Run with T, 3, 3
    epsilon = 1e-12
    _TP = []
    _TN = []
    _FN = []
    _FP = []
    for i in range(1, conf_mat.shape[1]):
        _TP.append(conf_mat[:, i:, i:].sum((1,2)))
        _TN.append(conf_mat[:, :i, :i].sum((1,2)))
        _FN.append(conf_mat[:, :i, i:].sum((1,2)))
        _FP.append(conf_mat[:, i:, :i].sum((1,2)))
        if aggregate:
            _TP[-1] = _TP[-1].sum()
            _TN[-1] = _TN[-1].sum()
            _FN[-1] = _FN[-1].sum()
            _FP[-1] = _FP[-1].sum()
        _FP[-1] = _FP[-1]+epsilon
        _FN[-1] = _FN[-1]+epsilon
    
    ret = {f'F1_{i}': (2 * _TP[i]) / (2 * _TP[i] + _FN[i] + _FP[i]) for i in range(0, len(_TP))}    
    ret.update({f'CSI_{i}': (_TP[i]) / (_TP[i] + _FN[i] + _FP[i]) for i in range(0, len(_TP))})
    ret.update({f'POD_{i}': _TP[i] / (_TP[i] + _FN[i]) for i in range(0, len(_TP))})
    ret.update({f'FAR_{i}': _FP[i] / (_TP[i] + _FP[i]) for i in range(0, len(_TP))})
    ret.update({f'BIAS_{i}': (_TP[i] + _FP[i]) / (_TP[i] + _FN[i]) for i in range(0, len(_TP))})
    return ret
    
    # _TP1 = conf_mat[:, 1:, 1:].sum((1,2))
    # _TN1 = conf_mat[:, :1, :1].sum((1,2))
    # _FN1 = conf_mat[:, :1, 1:].sum((1,2))
    # _FP1 = conf_mat[:, 1:, :1].sum((1,2))
    # _TP2 = conf_mat[:, 2:, 2:].sum((1,2))
    # _TN2 = conf_mat[:, :2, :2].sum((1,2))
    # _FN2 = conf_mat[:, :2, 2:].sum((1,2))
    # _FP2 = conf_mat[:, 2:, :2].sum((1,2))
    # # to prevent zero division error
    # if aggregate:
    #     _TP1 = _TP1.sum()
    #     _TN1 = _TN1.sum()
    #     _FN1 = _FN1.sum()
    #     _FP1 = _FP1.sum()
    #     _TP2 = _TP2.sum()
    #     _TN2 = _TN2.sum()
    #     _FN2 = _FN2.sum()
    #     _FP2 = _FP2.sum()
    # _FP1 = _FP1+epsilon
    # _FP2 = _FP2+epsilon
    # _FN1 = _FN1+epsilon
    # _FN2 = _FN2+epsilon
    # return {'F1_1': (2 * _TP1) / (2 * _TP1 + _FN1 + _FP1),
    #         'F1_2': (2 * _TP2) / (2 * _TP2 + _FN2 + _FP2),
    #         'CSI_1': (_TP1) / (_TP1 + _FN1 + _FP1),
    #         'CSI_2': (_TP2) / (_TP2 + _FN2 + _FP2),
    #         'POD_1': _TP1 / (_TP1 + _FN1),
    #         'POD_2': _TP2 / (_TP2 + _FN2),
    #         'FAR_1': _FP1 / (_TP1 + _FP1),
    #         'FAR_2': _FP2 / (_TP2 + _FP2),
    #         'BIAS_1': (_TP1 + _FP1) / (_TP1 + _FN1),
    #         'BIAS_2': (_TP2 + _FP2) / (_TP2 + _FN2),
    #         }

@torch.no_grad()
def satellite_evaluate(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, interval_point=(0.1, 1., 10.)):
    '''
    gt, pred: Both are 5D (BCTHW) Tensors.
              If shapes are different, bilinear upsample will be applied to pred.
              Assume `zr_relation` is already applied to both tensors.
    interval: List of numbers. [1,10] for default (x<1, 1<=x<10, 10<=x)
    mask:     Mask used to indicate out-of-range area. In is 0, Out is 1.
    '''
    mask = ~mask  # Now mask==1 denotes valid area
    device = 'cuda'

    T = gt.shape[2]
    gt = rearrange(gt, 'b c t h w -> t (b c h w)').to(device)
    pred = rearrange(pred, 'b c t h w -> t (b c h w)').to(device)
    mask = rearrange(mask, 'b c t h w -> t (b c h w)').to(device)

    # assert len(interval_point) == 2, 'currently only valid for interval with 2 points'
    num_intervals = len(interval_point) + 1
    interval_point = torch.tensor(interval_point).to(device=device)

    # Assume same shape tensors.
    assert pred.shape == gt.shape, f'shape of gt is {gt.shape}, shape of pred is {pred.shape}'

    # Transform to classification results
    gt_class = torch.bucketize(gt, interval_point, right=True)
    pred_class = torch.bucketize(pred, interval_point, right=True)

    # conf_mat: (T, pred, gt)
    conf_mat = torch.zeros(T, num_intervals, num_intervals)
    for t in range(T):
        for idx_pred in range(num_intervals):
            for idx_gt in range(num_intervals):
                conf_mat[t, idx_pred, idx_gt] += ((pred_class[t][mask[t]] == idx_pred)
                                               & (gt_class[t][mask[t]] == idx_gt)).float().sum().cpu()

    return conf_mat

def batch_scores_from_conf_mat(conf_mat: torch.Tensor) -> dict:
    # Run with B, T, 3, 3
    epsilon = 1e-12
    _TP1 = conf_mat[:, :, 1:, 1:].sum((3,2))
    _TN1 = conf_mat[:, :, :1, :1].sum((3,2))
    _FN1 = conf_mat[:, :, :1, 1:].sum((3,2))
    _FP1 = conf_mat[:, :, 1:, :1].sum((3,2))
    _TP2 = conf_mat[:, :, 2:, 2:].sum((3,2))
    _TN2 = conf_mat[:, :, :2, :2].sum((3,2))
    _FN2 = conf_mat[:, :, :2, 2:].sum((3,2))
    _FP2 = conf_mat[:, :, 2:, :2].sum((3,2))
    # to prevent zero division error
    _FP1 = _FP1+epsilon
    _FP2 = _FP2+epsilon
    _FN1 = _FN1+epsilon
    _FN2 = _FN2+epsilon
    F1_2 = (2 * _TP2) / (2 * _TP2 + _FN2 + _FP2)
    score = -(_FP2 + _FN2) / (_TP2 + 1e-3)
    return F1_2, score

@torch.no_grad()
def batch_satellite_evaluate(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, interval_point=(0.1, 1., 10.)):
    '''
    gt, pred: Both are 5D (BCTHW) Tensors.
              If shapes are different, bilinear upsample will be applied to pred.
              Assume `zr_relation` is already applied to both tensors.
    interval: List of numbers. [1,10] for default (x<1, 1<=x<10, 10<=x)
    mask:     Mask used to indicate out-of-range area. In is 0, Out is 1.
    '''
    mask = ~mask  # Now mask==1 denotes valid area
    device = 'cuda'

    B = gt.shape[0]
    T = gt.shape[2]
    gt = rearrange(gt, 'b c t h w -> b t (c h w)').to(device)
    pred = rearrange(pred, 'b c t h w -> b t (c h w)').to(device)
    mask = rearrange(mask, 'b c t h w -> b t (c h w)').to(device)

    # assert len(interval_point) == 2, 'currently only valid for interval with 2 points'
    num_intervals = len(interval_point) + 1
    interval_point = torch.tensor(interval_point).to(device=device)

    # Assume same shape tensors.
    assert pred.shape == gt.shape, f'shape of gt is {gt.shape}, shape of pred is {pred.shape}'

    # Transform to classification results
    gt_class = torch.bucketize(gt, interval_point, right=True)
    pred_class = torch.bucketize(pred, interval_point, right=True)

    # conf_mat: (T, pred, gt)
    conf_mat = torch.zeros(B, T, num_intervals, num_intervals)
    for b in range(B):
        for t in range(T):
            for idx_pred in range(num_intervals):
                for idx_gt in range(num_intervals):
                    conf_mat[b, t, idx_pred, idx_gt] += ((pred_class[b,t][mask[b,t]] == idx_pred)
                                                   & (gt_class[b,t][mask[b,t]] == idx_gt)).float().sum().cpu()

    return conf_mat

def batch_parse_time(input:List[List[str]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
    '''
    Here we parse 2d list str with shape T x B.
    Returns: B x T LongTensor of monthday idx(0~364) and B x T LongTensor of hour idx(0~23)
    '''
    T = len(input)
    B = len(input[0])
    monthday_idx:List[List[int]] = []
    hour_idx:List[List[int]] = []
    for b in range(B):
        tmp = [datetime.strptime(input[t][b], "%Y%m%d%H%M") for t in range(T)]
        hour_idx.append([day.hour for day in tmp]) # No need to process other thing.
        baseday = datetime(year=tmp[0].year, month=1, day=1)
        # change to day interval between Jan.1st
        tmp = [(day-baseday).days for day in tmp]
        # To handle leap year, subtract 1day if the sequence contains Dec.31st of leap year
        if 365 in tmp:
            tmp = [(day-1) % 365 for day in tmp]
        monthday_idx.append(tmp)
    hour_idx = torch.LongTensor(hour_idx)
    monthday_idx = torch.LongTensor(monthday_idx)
    return monthday_idx, hour_idx

if __name__ == '__main__':
    test = [['201904052320', '201904051520', '202012311210', '202012302350'],
            ['201904052330', '201904051530', '202012311220', '202012310000'],
            ['201904052340', '201904051540', '202012311230', '202012310010'],
            ['201904052350', '201904051550', '202012311240', '202012310020'],
            ['201904060000', '201904051600', '202012311250', '202012310030'],
            ['201904060010', '201904051610', '202012311300', '202012310040'],
            ['201904060020', '201904051620', '202012311310', '202012310050'],
            ['201904060120', '201904051720', '202012311410', '202012310150'],
            ['201904060220', '201904051820', '202012311510', '202012310250'],
            ['201904060320', '201904051920', '202012311610', '202012310350'],
            ['201904060420', '201904052020', '202012311710', '202012310450'],
            ['201904060520', '201904052120', '202012311810', '202012310550'],
            ['201904060620', '201904052220', '202012311910', '202012310650']]
    parsing_ret = batch_parse_time(test)
    print(parsing_ret[0], parsing_ret[0].shape)
    print(parsing_ret[1], parsing_ret[1].shape)
