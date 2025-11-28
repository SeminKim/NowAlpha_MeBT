from PIL import Image
from tats.utils import f1_csi_from_conf_mat, batch_satellite_evaluate
from multiprocessing import Pool
import copy
import cv2
from tats.data import zr_relation, normalize_dBZ, unnormalize_dBZ
from torch.utils.data import DataLoader
import netCDF4 as nc
from torch.nn import functional as F
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
from tqdm import tqdm
import importlib
from omegaconf import OmegaConf
from taming.data.utils import channel_to_spatial as c2s
from tats.data import SatelliteDataset4VQGAN_infer, normalize_dBZ
import os
import argparse
import pytorch_lightning as pl
from glob import glob

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

matplotlib.rcParams['figure.dpi'] = 300


torch.set_grad_enabled(False)
pl.seed_everything(42)

color = ['#FAFAFA', '#00C8FF', '#009BF5', '#004AF5', '#00FF00', '#00BE00', '#008C00', '#005A00', '#FFFF00', '#FFDC1F', '#F9CD00', '#E0B900',
         '#CCAA00', '#FF6600', '#FF3200', '#D20000', '#B40000', '#E0A9FF', '#C969FF', '#B329FF', '#9300E4', '#B3B4DE', '#4C4EB1', '#000390', '#333333']  # new
cla_num = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # new
cls_thres = [0.09, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 90, 110, 150, 200]  # new
cmap = matplotlib.colors.ListedColormap(color)


def s2c(img, size=[4, 4]):
    img = rearrange(img, 'b c t (h H) (w W) -> b (c H W) t h w', H=size[0], W=size[1])
    return img


@torch.no_grad()
def compute_optical_flow(result, ignores=['samples']):
    masks = result['gt'] < 0.
    opt_dict = {}
    for k, pred in result.items():
        all_of = []
        for b, v in enumerate(pred):
            v = v.cpu()
            if k in ignores:
                continue
            T, H, W = v.shape
            optflow = np.zeros([T, H, W, 2])
            for t in range(1, len(optflow)):
                mask = torch.logical_or(masks[b, t-1], masks[b, t])
                curr = torch.where(mask, torch.zeros_like(v[t]), v[t]).numpy()
                past = torch.where(mask, torch.zeros_like(v[t-1]), v[t-1]).numpy()
                flow_fn = cv2.optflow.createOptFlow_PCAFlow()
                try:
                    of = flow_fn.calc(past, curr, None)
                    of = np.where(mask.unsqueeze(-1), np.zeros_like(of), of)
                    optflow[t] = of
                except Exception as e:
                    optflow[t] = 0.
            all_of.append(optflow)
        all_of = np.stack(all_of)
        opt_dict[k] = all_of
    return opt_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model(config, ckpt_path):
    args = argparse.Namespace()
    args.base = config
    args.ckpt_path = ckpt_path

    config = OmegaConf.load(args.base)
    config.model.params.class_cond_dim = None
    config.model.vqvae.target = 'tats.tats_vqgan.VQGAN'
    config.model.vqvae.params.spatial_to_channel = [4, 4]
    try:
        config.model.params.latent_shape = config.data.latent_shape
    except:
        config.model.params.latent_shape = config.model.mask.params.shape
    model: pl.LightningModule = get_obj_from_str(config.model["target"])(
        config.model.params, config.model.vqvae, config.model.mask, cond_stage_key=config.model.params.cond_stage_key)
    # load the most recent checkpoint file
    vq_model = None
    # if config.model['target'].split('.')[-1] == 'Net2NetTransformerVTokenEval_2StagesInference':
    vq_model = copy.deepcopy(model.first_stage_model)
    print('will start from the recent ckpt %s' % args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    global_step = ckpt['global_step']
    context_len = config.data.before // 10
    vid_t = model.t_lengths[context_len:]
    prior_t = model.t_prior(vid_t, global_step)
    config.time_limit = (prior_t.argmax().item() + 1) * 10
    m, u = model.load_state_dict(ckpt['state_dict'], strict=False)
    if vq_model is not None:
        model.first_stage_model = vq_model
    for k in u:
        if 'first_stage_model' in k:
            pass
        else:
            print(k)
            assert 0

    return model, config


def f1_csi_from_conf_mat(conf_mat: torch.Tensor) -> dict:
    epsilon = 1e-5
    if len(conf_mat.shape) == 4:
        conf_mat = conf_mat.sum(0)
    _TP1 = conf_mat[:, 1:, 1:].sum((1, 2)).cpu().numpy()
    _TN1 = conf_mat[:, :1, :1].sum((1, 2)).cpu().numpy()
    _FN1 = conf_mat[:, :1, 1:].sum((1, 2)).cpu().numpy()
    _FP1 = conf_mat[:, 1:, :1].sum((1, 2)).cpu().numpy()
    _TP2 = conf_mat[:, 2:, 2:].sum((1, 2)).cpu().numpy()
    _TN2 = conf_mat[:, :2, :2].sum((1, 2)).cpu().numpy()
    _FN2 = conf_mat[:, :2, 2:].sum((1, 2)).cpu().numpy()
    _FP2 = conf_mat[:, 2:, :2].sum((1, 2)).cpu().numpy()
    # to prevent zero division error
    _FP1 = _FP1+epsilon
    _FP2 = _FP2+epsilon
    _FN1 = _FN1+epsilon
    _FN2 = _FN2+epsilon
    return {'F1_1': (2 * _TP1) / (2 * _TP1 + _FN1 + _FP1),
            'F1_2': (2 * _TP2) / (2 * _TP2 + _FN2 + _FP2),
            'CSI_1': (_TP1) / (_TP1 + _FN1 + _FP1),
            'CSI_2': (_TP2) / (_TP2 + _FN2 + _FP2),
            'POD_1': _TP1 / (_TP1 + _FN1),
            'POD_2': _TP2 / (_TP2 + _FN2),
            'FAR_1': _FP1 / (_TP1 + _FP1),
            'FAR_2': _FP2 / (_TP2 + _FP2),
            'BIAS_1': (_TP1 + _FP1) / (_TP1 + _FN1),
            'BIAS_2': (_TP2 + _FP2) / (_TP2 + _FN2),
            }


def continuous_draw2(m, x, y, result, datestr, out_dir, mode='pred', time=None, postfix='_cont2', of=None, interval=20):
    os.makedirs(os.path.join(out_dir, datestr), exist_ok=True)
    gt_cls = np.zeros_like(result)
    for i, ct in enumerate(cls_thres):
        gt_cls[result >= ct] = i+1
    fig = plt.figure(figsize=(8, 8))
    cs = m.contourf(x, y, gt_cls, levels=cla_num, cmap=cmap, zorder=1)
    # cbar = m.colorbar(cs, location='right', pad='5%')
    # cbar.set_ticks(cla_num) #9
    # cbar.set_ticklabels(['','0','0.1','0.5','1','2','3','4','5','6','7','8','9','10','15','20','25','30','40','50','60','70','90','110','150','200']) #new
    # cbar.ax.tick_params(labelsize=15)

    m.drawcoastlines(linewidth=1.0, zorder=3,)
    m.drawcountries(linewidth=1.0, zorder=3,)
    parallels = np.arange(30., 46, 2.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=15)
    meridians = np.arange(120., 134., 2.)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=15)

    # set title
    yy = datestr[:4]
    mm = datestr[4:6]
    dd = datestr[6:8]
    hh = datestr[8:10]
    mi = datestr[10:12]
    datestr_ = f'{yy}-{mm}-{dd} {hh}:{mi}'
    if time >= 0:
        title = f'{datestr_} KST +{(time+1)*interval}min'
    else:
        title = f'{datestr_} KST {(time+1)*interval}min'
    plt.title(title, fontsize=20)

    if of is not None:
        u = of[:, :, 0]
        v = of[:, :, 1]
        stride = 40
        m.quiver(x[::stride, ::stride], y[::stride, ::stride], u[::stride, ::stride], v[::stride, ::stride], scale=100)
        postfix += '_of'

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, datestr, f'{time*interval}min_{mode}{postfix}.png'), dpi=150)
    print(os.path.join(out_dir, datestr, f'{time*interval}min_{mode}{postfix}.png'))
    plt.close()


def get_latlon():
    print(" >>> get coordinates from latlon.npy, latlon_context.npy...")
    # load map
    m = Basemap(projection='merc', lat_0=38, lon_0=126, resolution='h',
                urcrnrlat=44, llcrnrlat=30, llcrnrlon=120, urcrnrlon=134)

    latlon = np.load('./visualize/latlon.npy')

    lats = latlon[:, :, 1]
    lons = latlon[:, :, 0]
    x, y = m(lons, lats)

    # get whole HSR reference
    latlon_context = np.load('./visualize/latlon_context.npy')
    lats_context = latlon_context[:, :, 1]
    lons_context = latlon_context[:, :, 0]

    x_context, y_context = m(lons_context, lats_context)
    return m, x, y, lats, lons
    # return lats, lons


def generate_nc(nc_out_dir, datestr, lats, lons, result, context=7, interval=20, after=180, postfix=''):
    '''
        for given target time, perform predictions and save them into nc file
    '''

    nc_out_path = os.path.join(nc_out_dir, f'RNN_precipitation_{datestr}{postfix}.nc')
    os.makedirs(nc_out_dir, exist_ok=True)
    nc_dataset = nc.Dataset(nc_out_path, mode='w', format='NETCDF4')

    complevel = 5
    x_size, y_size = 1152, 1440

    total_len = context + after//10

    X_dim = nc_dataset.createDimension('nx', x_size)
    Y_dim = nc_dataset.createDimension('ny', y_size)
    T_dim = nc_dataset.createDimension('nz', total_len)
    dt = nc_dataset.createDimension('dt', 600)  # 600s = 10min

    # variables
    X_var = nc_dataset.createVariable('nx', np.float64, ('nx',), zlib=True,
                                      complevel=complevel)
    X_var[:] = np.arange(1., float(x_size + 1))
    X_var.units = 'units'
    X_var.long_name = 'X'
    Y_var = nc_dataset.createVariable('ny', np.float64, ('ny',), zlib=True,
                                      complevel=complevel)
    Y_var[:] = np.arange(1., float(y_size + 1))
    Y_var.units = 'units'
    Y_var.long_name = 'Y'

    # variables - latlons
    lon_var = nc_dataset.createVariable('longitude', np.float32, ('ny', 'nx'), zlib=True,
                                        complevel=complevel)
    lon_var[:, :] = lons
    lon_var.units = 'deg'

    lat_var = nc_dataset.createVariable('latitude', np.float32, ('ny', 'nx'), zlib=True,
                                        complevel=complevel)
    lat_var[:, :] = lats
    lat_var.units = 'deg'

    # a group for predictions per each lead time
    preds = nc_dataset.createVariable('RAIN', np.float32, ('nz', 'ny', 'nx'), zlib=True,
                                      complevel=complevel)

    assert result.shape[0] == total_len, f"result shape is {result.shape}"
    preds.units = 'mm/hr'
    preds.long_name = 'Precipitation'
    preds.coordinates = 'longitude latitude'
    preds[:, :, :] = result[-total_len:, :, :]
    print(preds.shape)

    nc_dataset.close()


@torch.inference_mode()
def build_tf_input(ret, model, time_limit=None, interval=10):
    vq_model = model.first_stage_model
    ret['gts'] = ret['video'].cuda()
    ret['mask'] = ret['mask'].cuda()
    ret['input_indices'] = ret['input_indices'].cuda()
    ret['pred_indices'] = ret['pred_indices'].cuda()
    NC = ret['input_indices'].shape[1] // model.num_pos
    NP = ret['pred_indices'].shape[1] // model.num_pos
    T = NP + NC
    B, C, _, H, W = ret['gts'].shape

    # encode interior video
    vq_input = torch.zeros(B, C, T, H, W, device=model.device)
    vq_input[:, :, :NC] = ret['gts'][:, :, :NC]  # replicate padding
    vq_input[:, :, NC:] = ret['gts'][:, :, NC-1:NC]  # replicate padding

    vid = normalize_dBZ(vq_input)  # shape with N C T H W
    vq_vid = vq_model.encode(vid)
    vq_vid = vq_vid.long()
    ret['interior_video'] = vq_vid

    ret['inputs'] = ret['video'] = vq_model.encode(normalize_dBZ(ret['gts'])).long()
    ret['gts'] = rearrange(ret['gts'], 'b c t h w -> b t h w c')

    if time_limit is not None:
        frames_limit = time_limit // interval
        _, _, h, w = vq_vid.shape
        ret['pred_indices'] = ret['pred_indices'][:, :frames_limit*h*w]
        frames_limit = frames_limit + ret['input_indices'].shape[1] // (h*w)
        ret['interior_video'] = ret['interior_video'][:, :frames_limit]
        ret['video'] = ret['video'][:, :frames_limit]
        ret['gts'] = ret['gts'][:, :frames_limit]
        ret['eval_idx'] = ret['eval_idx'][ret['eval_idx'] < frames_limit]
    return ret


@torch.inference_mode()
def build_revise_input(ret, config, time_limit, config2, time_limit2):
    # TODO: update interior_video, input_indices, pred_indices
    context_len1 = config.data.before // 10
    context_len2 = config2.data.before // 10
    start_idx = context_len1 - context_len2
    assert start_idx >= 0

    context_len = config2.data.before // 10
    pred_len = time_limit2 // 10

    input_indices = torch.arange(context_len * 256)
    pred_indices = torch.arange(context_len * 256, (context_len + pred_len) * 256)
    B = ret['video'].shape[0]
    device = ret['video'].device

    ret['input_indices'] = input_indices.unsqueeze(0).repeat(B, 1).to(device)
    ret['pred_indices'] = pred_indices.unsqueeze(0).repeat(B, 1).to(device)
    ret['interior_video'] = ret['interior_video'][:, start_idx:start_idx+context_len+pred_len]

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/satellite/interior_target_maskgit_v2_longest_3hrs.yaml')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/maskgit_3hrs_250k.ckpt')
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--sampling_temperature', type=float, default=1.)
    parser.add_argument('--mask_temperature', type=float, default=4.5)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--save_nc', action='store_true')
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--save_of', action='store_true')
    parser.add_argument('--ignore', nargs='+', default=['avg', 'worst_llh', 'max', 'worst', 'best_llh', 'random'])
    parser.add_argument('--target_dates', default='20240712_target_dates_all.txt', type=str)
    parser.add_argument('--after', type=int, default=360)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--time_limit', type=int, default=None)
    parser.add_argument('--n_stage', action='store_true')
    parser.add_argument('--n_stage_v2', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--config2', default=None)
    parser.add_argument('--ckpt_path2', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--t_revise', type=int, default=0)
    parser.add_argument('--M', type=int, default=0)
    parser.add_argument('--revise_temp', type=float, default=1.0)
    parser.add_argument('--pass_if_exist', action='store_true')
    args = parser.parse_args()

    # args.n_stage = True
    # args.n_stage_v2 = True
    if args.train:
        args.target_dates = "0712_train_target_dates.txt"
    with open(args.target_dates, 'r') as f:
        target_dates = f.readlines()
    target_dates = [t.strip() for t in target_dates]
    target_dates = list(sorted((target_dates)))

    if args.pass_if_exist:
        exists = []
        fnames = glob(f"nc_results/{args.exp}/metrics/*_metrics_pred_z.npy")
        exists = [os.path.basename(f).split('_')[0] for f in fnames]
        print("Passing if exists")
        print("Given target dates: ", len(target_dates))
        target_dates = [t for t in target_dates if t not in exists]
        print("After passing: ", len(target_dates))

    if args.debug:
        target_dates = target_dates[:1]
        args.exp = 'debug'
        args.no_vis = True

    print(f'will generate {len(target_dates)} nc files')
    model, config = load_model(args.config, args.ckpt_path)
    if args.config2 is not None:
        model2, config2 = load_model(args.config2, args.ckpt_path2)
        model2 = model2.cuda()
        time_limit2 = config2.time_limit
    else:
        model2 = None
    config.data.latent_shape = config.model.mask.params.shape
    config_data = config.data
    config_data['batch_size'] = 1
    config.data.after = config_data.after = args.after  # default 360min. (6hr)
    model = model.cuda()
    model.eval()

    config_data.eval_t = list(range(-config.data.before, config.data.after, 10))

    # TODO: pull it from config
    interval = config.data.input_interval
    before = config.data.before
    context_len = before // interval
    after = config.data.after
    stride = 60 // interval
    time_limit = config.time_limit
    pred_len = after // interval

    m, x, y, lats, lons = get_latlon()
    tg_h, tg_w = 1440, 1152
    base = f"nc_results/{args.exp}"
    pickle_dir = f'sal_pickles/{args.exp}'
    vis_out_dir = f'vis_results/{args.exp}'
    metrics_out_dir = f'{base}/metrics'

    nc_out_dir = base
    os.makedirs(vis_out_dir, exist_ok=True)
    os.makedirs(nc_out_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(metrics_out_dir, exist_ok=True)

    # build dataset
    dataset = SatelliteDataset4VQGAN_infer(
        'datasets/HSR_dBZ',
        target_dates,
        config_data.data_interval,
        config_data.input_interval,
        config_data.before,
        config_data.output_interval,
        config_data.after,
        config_data.latent_shape,
        load_future=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for b_idx, batch in tqdm(enumerate(loader), total=len(loader.dataset)):
        # img_out_dict_all = {'gt': [], 'best': [], 'pred_z': []}
        img_out_dict_all = {'gt': [], 'pred_z': []}
        gt = {}
        pred = {}
        conf_mats = {}
        kwargs = dict(
            n_steps=args.n_steps,
            sampling_temperature=args.sampling_temperature,
            mask_temperature=args.mask_temperature,
            teacher_forcing=False,
            return_log_likelihood=True,
            time_limit=time_limit,
            fix_context=False,
        )
        if not args.n_stage:
            batch = build_tf_input(batch, model, time_limit=None)
            recon = model.first_stage_model.decode(batch['video'].cuda())
            _, img_out_dict, _ = model.gen_prediction(batch, **kwargs)
            img_out_dict['recon'] = c2s(unnormalize_dBZ(recon))
            img_out_dict['pred_z'] = c2s(unnormalize_dBZ(model.first_stage_model.decode(img_out_dict['pred_z'].cuda())))
            # 12 + time_limit // 10 frames.
            # conf_mat_dict, img_out_dict, _ = model.gen_prediction(batch, **kwargs)
        else:
            model.load_dBZ = False
            batch = build_tf_input(batch, model, time_limit=None)
            recon = model.first_stage_model.decode(batch['video'].cuda())
            total_T = args.after  # hard coded for 6hr prediction
            time_limit = time_limit if time_limit is not None else args.after

            ################ DRAFT PHASE ################
            for t in range(0, total_T, time_limit):
                _, img_out_dict, _ = model.gen_prediction(batch, **kwargs)
                for k in img_out_dict_all:
                    img_out_dict_all[k].append(img_out_dict[k])
                batch['video'] = img_out_dict['pred_z'].cuda()
                kwargs['fix_context'] = True
            for k in img_out_dict_all:
                if k != 'pred_z':
                    img_out_dict_all[k] = torch.cat([x if i == 0 else x[:, :, -time_limit//interval:]
                                                    for i, x in enumerate(img_out_dict_all[k])], dim=2)[:, :, :context_len+pred_len]
                else:
                    img_out_dict_all[k] = torch.cat([x if i == 0 else x[:, -time_limit//interval:]
                                                    for i, x in enumerate(img_out_dict_all[k])], dim=1)[:, :context_len+pred_len]
            ################ DRAFT PHASE ################

            ################ REVISE PHASE ###############
            if args.config2 is not None:
                start_idx = -(args.after+config2.data.before) // interval  # 2hrs + 6hrs
                token_len = (config2.data.before + time_limit2) // interval
                token_copy = img_out_dict_all['pred_z'].clone()
                revise_context = True
                rev_batch = build_revise_input(batch, config, time_limit, config2, time_limit2)
                while True:
                    stop_flag = start_idx >= -token_len
                    if stop_flag:
                        revise_tokens = token_copy[:, -token_len:]
                    else:
                        revise_tokens = token_copy[:, start_idx:start_idx+token_len]
                    assert revise_tokens.shape[1] == token_len
                    rev_batch['video'] = revise_tokens.cuda()

                    # TODO: define build_revise_input
                    revise_tokens = model.revise(batch, args.t_revise, args.M, args.revise_temp, revise_context=revise_context)
                    revise_context = False
                    if stop_flag:
                        token_copy[:, -token_len:] = revise_tokens
                    else:
                        token_copy[:, start_idx:start_idx+token_len] = revise_tokens
                    start_idx += time_limit2 // interval
                    start_idx = max(start_idx, -token_len)
                    if stop_flag:
                        break

                img_out_dict_all['draft_z'] = c2s(unnormalize_dBZ(model.first_stage_model.decode(img_out_dict_all['pred_z'].cuda())))
                img_out_dict_all['pred_z'] = c2s(unnormalize_dBZ(model.first_stage_model.decode(token_copy.cuda())))
            else:
                img_out_dict_all['pred_z'] = c2s(unnormalize_dBZ(model.first_stage_model.decode(img_out_dict_all['pred_z'].cuda())))
            img_out_dict_all['recon'] = c2s(unnormalize_dBZ(recon))
            img_out_dict = img_out_dict_all
        # Draw and
        _gt = rearrange(batch['gts'], 'b t h w c -> b c t h w')
        _gt = c2s(_gt)

        img_out_dict['gt'] = _gt
        for k, v in img_out_dict.items():
            B, C, T, H, W = v.shape
            v = rearrange(v, 'b c t h w -> (b t) c h w')
            v = F.interpolate(v, size=(tg_h, tg_w), mode='bilinear', align_corners=False)
            v = rearrange(v, '(b t) c h w -> b c t h w', b=B).squeeze(1).cpu()
            if k == 'gt':
                mask = v < -250
            if k != 'best':
                img_out_dict[k] = zr_relation(v)
            else:
                img_out_dict[k] = v

        conf_mat_batch = batch_satellite_evaluate(img_out_dict['gt'].unsqueeze(1), img_out_dict['pred_z'].unsqueeze(1), mask.unsqueeze(1))
        for datestr, conf_mat in zip(batch['datestr'], conf_mat_batch):
            metrics = f1_csi_from_conf_mat(conf_mat)
            np.save(f'{metrics_out_dir}/{datestr}_confmat_pred_z.npy', conf_mat.cpu().numpy())
            np.save(f'{metrics_out_dir}/{datestr}_metrics_pred_z.npy', metrics)

        if args.save_nc:
            for k, v in img_out_dict.items():
                for i, pred in enumerate(v):
                    generate_nc(nc_out_dir, batch['datestr'][i], lats, lons, pred,
                                context=context_len, interval=interval, after=after, postfix=f'_{k}')

        # compute optical flow
        if args.save_of:
            of_dict = compute_optical_flow(img_out_dict)

        if not args.no_vis:
            for k, v in img_out_dict.items():
                for i, pred in enumerate(v):
                    if args.save_of:
                        with Pool(8) as p:
                            p.starmap(continuous_draw2, [(m, x, y, pred[context_len+lead_t].cpu().numpy(), batch['datestr'][i], vis_out_dir,
                                      k, lead_t, '_cont', of_dict[k][i][context_len + lead_t], interval) for lead_t in range(-context_len, pred_len)])
                    with Pool(8) as p:
                        p.starmap(continuous_draw2, [(m, x, y, pred[context_len+lead_t].cpu().numpy(), batch['datestr'][i],
                                  vis_out_dir, k, lead_t, '_cont', None, interval) for lead_t in range(-context_len, pred_len)])
    print("Done!")
