import argparse
import os
import math

from network import Network
import torch


def points_loss_function(y_pred, y, vec):
    """Unused loss function"""
    ones = torch.ones(1, 20).cuda()
    vec = torch.cat([vec, ones], dim=0)

    res_pred = torch.matmul(torch.reshape(y_pred, [-1, 4, 4]), vec)
    res = torch.matmul(torch.reshape(y, [-1, 4, 4]), vec)

    res_pred = torch.divide(res_pred[:, :3, :], res_pred[:, 3, None, :] + 1e-7)
    res = torch.divide(res[:, :3, :], res[:, 3, None, :] + 1e-7)

    diff = (res - res_pred) ** 2
    out = torch.mean(torch.sqrt(torch.sum(diff, dim=1)))
    return out


def normalized_l2_loss(pred, gt, reduce=True):
    """
    Returns normalized L2 loss: ||pred - gt||^2 / ||gt||^2
    """
    norm = torch.sum(gt ** 2, dim=-1) + 1e-7
    loss = torch.sum((pred - gt) ** 2, dim=-1) / norm
    return torch.mean(loss) if reduce else loss


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-r', '--resume', type=int, default=None)
    parser.add_argument('-nw', '--workers', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-iw', '--input_width', type=int, default=516)
    parser.add_argument('-ih', '--input_height', type=int, default=386)
    parser.add_argument('-e', '--epochs', type=int, default=250)
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-bb', '--backbone', type=str, default='resnet34')
    parser.add_argument('-de', '--dump_every', type=int, default=0)
    parser.add_argument('-w', '--weight', type=float, default=0.1)
    parser.add_argument('-ns', '--noise_sigma', type=float, default=None)
    parser.add_argument('-ts', '--t_sigma', type=float, default=0.0)
    parser.add_argument('-rr', '--random_rot', action='store_true', default=False)
    parser.add_argument('-wp', '--weights_path', type=str, default=None)
    parser.add_argument('-vis', '--visualize', action='store_true', default=False)
    parser.add_argument('-mod', '--modifications', type=str, default=None)
    parser.add_argument('-mc', '--mc_samples', type=int, default=50)
    parser.add_argument('-dpt', '--dropout_prob_trans', type=float, default=0)
    parser.add_argument('-dpr', '--dropout_prob_rot', type=float, default=0)
    parser.add_argument('-dpb', '--dropout_prob_backbone', type=float, default=0.0, help='Dropout probability for layer after backbone')
    parser.add_argument('-sn', '--sample_nbr', type=int, default=3)
    parser.add_argument('-ccw', '--complexity_cost_weight', type=float, default=0.001)
    parser.add_argument('-bt', '--bayesian_type', type=int, default=0, help='Bayesian head type (0: all, 1: only first layer, 2: only last layer, 3: only middle layer)')
    parser.add_argument('-is', '--input_sigma', type=float, default=0.1)
    parser.add_argument('-bs', '--bootstrap_samples', type=int, default=0, help='Number of bootstrapped ensemble models, 0 to disable')
    parser.add_argument('-et', '--ensemble_type', type=int, default=1, help='Ensemble type (1: random weights + data shuffle, 2: data bootstrap + all from 1.. The model 3 is actually done like 2 + you add parameter -ns - noise sigma to add noise to the data.)')
    parser.add_argument('-ale', '--use_aleatoric', action='store_true', default=False, help="Whether to output aleatoric uncertainty estimates (kappa and sigma_t) during inference), will change loss completely.")
    parser.add_argument('--ensemble_dir', type=str, default=None, help="Directory with ensemble .pth models")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for inference results")
    parser.add_argument('path')

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


def remap_bayesian_state_dict(raw_sd, init_sigma=0.1, bayes_type=0):
    """Remap vanilla checkpoint to Bayesian format."""
    rho0 = math.log(math.exp(init_sigma) - 1.0)
    base_sd = {}

    for k, v in raw_sd.items():
        prefix = k.split('.')[0]
        idx = k.split('.')[1] if '.' in k else None
        if prefix in ('fc_z','fc_y','fc_t') and k.endswith(('weight','bias')) and \
           (bayes_type in {0, 4} or 
            (bayes_type == 1 and idx == '0') or
            (bayes_type == 2 and idx == '4') or
            (bayes_type == 3 and idx == '2')):
            base_sd[k + '_mu'] = v
        else:
            base_sd[k] = v

    full_sd = {}
    for k, v in base_sd.items():
        full_sd[k] = v
        if k.endswith('_mu'):
            base_key = k[:-3]
            full_sd[base_key + '_rho'] = torch.full_like(v, rho0)
            full_sd[f"{base_key}_sampler.mu"] = v.clone()
            full_sd[f"{base_key}_sampler.rho"] = torch.full_like(v, rho0)
            full_sd[f"{base_key}_sampler.eps_w"] = torch.randn_like(v)

    return full_sd


def remap_dropout_state_dict(base_sd, args):
    """
    Remap old baseline checkpoint to new mc_dropout backbone structure.
    - fc_z / fc_y / fc_t: keeps your original mapping rules
    - backbone layer blocks: block i → new index (2 * i)
    """
    new_sd = {}

    for k, v in base_sd.items():
        parts = k.split('.')

        # 1) HEAD REMAPPING
        if parts[0] in ('fc_z', 'fc_y', 'fc_t') and parts[1].isdigit():
            idx = int(parts[1])
            if idx in (2, 4):
                # old mapping
                new_idx = {2: 3, 4: 6}[idx]
                parts[1] = str(new_idx)
                new_sd['.'.join(parts)] = v
                continue

        new_sd[k] = v

    return new_sd

def remap_aleatoric_state_dict(base_sd, args):
    new_sd = {}

    for k, v in base_sd.items():

        # fc_y: 3 → 4
        if k == "fc_y.4.weight":
            new_w = torch.zeros(4, v.shape[1])
            new_w[:3] = v
            torch.nn.init.xavier_uniform_(new_w[3:])
            new_sd[k] = new_w
            continue

        if k == "fc_y.4.bias":
            new_b = torch.zeros(4)
            new_b[:3] = v
            new_sd[k] = new_b
            continue

        # fc_t: 3 → 6
        if k == "fc_t.4.weight":
            new_w = torch.zeros(6, v.shape[1])
            new_w[:3] = v
            torch.nn.init.xavier_uniform_(new_w[3:])
            new_sd[k] = new_w
            continue

        if k == "fc_t.4.bias":
            new_b = torch.zeros(6)
            new_b[:3] = v
            new_sd[k] = new_b
            continue

        # everything else
        new_sd[k] = v

    return new_sd

def load_model(args):
    model = Network(args).cuda()

    if args.weights_path is not None:
        print("Loading weights from:", args.weights_path)
        raw_sd = torch.load(args.weights_path, map_location='cpu', weights_only=True)

        if args.modifications == "mc_dropout":
            state_dict = remap_dropout_state_dict(raw_sd, args)
        elif args.modifications == "bayesian":
            state_dict = remap_bayesian_state_dict(raw_sd, init_sigma=args.input_sigma, bayes_type=args.bayesian_type)
        elif args.use_aleatoric:
            if any("fc_y.4.weight" in k and raw_sd[k].shape[0] > 3 for k in raw_sd):
                state_dict = raw_sd
            else:
                state_dict = remap_aleatoric_state_dict(raw_sd, args)
        else:
            state_dict = raw_sd

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

    if args.resume is not None:
        sd_path = f'checkpoints/{args.resume:03d}.pth'
        print("Resuming from:", sd_path)
        model.load_state_dict(torch.load(sd_path), strict=True)

    return model

def load_models(args):
    if args.ensemble_dir is None:
        return [load_model(args).eval()]

    models = []
    for ckpt in sorted(os.listdir(args.ensemble_dir)):
        if not ckpt.endswith(".pth"):
            continue

        ckpt_path = os.path.join(args.ensemble_dir, ckpt)
        args.weights_path = ckpt_path   # reuse existing loader
        model = load_model(args).eval()
        models.append(model)

    if not models:
        raise RuntimeError("Empty ensemble directory")

    return models


