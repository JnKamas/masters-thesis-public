import os
import torch
import numpy as np
from dataset import Dataset
from network import Network
from network_helpers import  parse_command_line, load_model
from torch.utils.data import DataLoader
from shutil import copyfile
from tqdm.auto import tqdm

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def build_transform(z, y, t, eps=1e-8):
    z = z / (np.linalg.norm(z) + eps)
    y = y - np.dot(z, y) * z
    y = y / (np.linalg.norm(y) + eps)
    x = np.cross(y, z)

    T = np.zeros((4,4), dtype=np.float32)
    T[:3,0] = x
    T[:3,1] = y
    T[:3,2] = z
    T[:3,3] = t
    T[3,3] = 1.0
    return T

def save_prediction(path, transform, kappa_i=None, sigma_i=None):
    with open(path, "w") as f:
        np.savetxt(
            f,
            transform.T.ravel()[None, :],
            fmt="%1.6f",
            newline=" "
        )
        f.write("\n")

        if kappa_i is not None:
            f.write("# kappa\n")
            f.write(f"{kappa_i:.6f}\n")

        if sigma_i is not None:
            f.write(
                "# sigma_tx sigma_ty sigma_tz\n"
                f"{sigma_i[0]:.6f} {sigma_i[1]:.6f} {sigma_i[2]:.6f}\n"
            )

def infer(args, export_to_folder=True):
    # load and set eval()
    model = load_model(args).eval()

    # figure out dataset root
    dir_path = os.path.dirname(args.path)

    # derive model name from your checkpoint arg
    weights_path = getattr(args, 'weights', None) \
                or getattr(args, 'weights_path', None) \
                or getattr(args, 'resume', None)
    if args.modifications == "ensemble" and weights_path:
        # models/ensemble1/modelA.pth → ensemble1
        model_name = os.path.basename(os.path.dirname(weights_path))
    elif weights_path:
        model_name = os.path.splitext(os.path.basename(weights_path))[0]
    else:
        model_name = 'model'

    # root for all predictions: ~/inference/<model_name>
    if args.out_dir is not None:
        export_root = os.path.expanduser(args.out_dir)
    else:
        export_root = os.path.expanduser(f'~/thesis/inference/{model_name}')
    os.makedirs(export_root, exist_ok=True)

    # prepare data
    val_dataset = Dataset(
        args.path,
        'val',
        args.input_width,
        args.input_height,
        preload=not args.no_preload
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        if args.modifications == "mc_dropout":
            model.eval()
            enable_dropout(model)

        PRINT_PREDS = True   # set True to print GT + predictions

        progress = tqdm(
            val_loader,
            desc="Running Inference",
            ncols=80,
            dynamic_ncols=True,
            ascii=False,
            bar_format=(
                "{l_bar}{bar} {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
        )

        for sample in progress:

            if args.modifications in {"mc_dropout", "bayesian"}:
                n_passes = args.mc_samples
            else:
                n_passes = 1

            for mc_idx in range(n_passes):
                if args.use_aleatoric:
                    z, y, t, s_R, s_t = model(sample['xyz'].cuda())
                else:
                    z, y, t = model(sample['xyz'].cuda())

                pred_zs = z.cpu().numpy()
                pred_ys = y.cpu().numpy()
                pred_ts = t.cpu().numpy()

                if args.use_aleatoric:
                    sigma_R = torch.nn.functional.softplus(s_R)
                    sigma_t = torch.nn.functional.softplus(s_t)
                    kappa = 1.0 / (sigma_R**2 + 1e-8)

                    pred_kappas = kappa.cpu().numpy()
                    pred_sigma_ts = sigma_t.cpu().numpy()
                else:
                    pred_kappas = None
                    pred_sigma_ts = None

                for i in range(len(pred_zs)):
                    transform = build_transform(pred_zs[i], pred_ys[i], pred_ts[i])

                    txt_path = sample['txt_path'][i].replace("\\","/")
                    subdir = os.path.dirname(txt_path)

                    dst = os.path.join(export_root, subdir)
                    os.makedirs(dst, exist_ok=True)

                    if n_passes > 1:
                        base = os.path.basename(txt_path).replace(".txt","")
                        name = f"prediction{mc_idx}_{base}.txt"
                    else:
                        name = f"prediction_{os.path.basename(txt_path)}"

                    out_file = os.path.join(dst, name)

                    if args.use_aleatoric:
                        kappa_i = pred_kappas[i, 0]
                        sigma_i = pred_sigma_ts[i]
                    else:
                        kappa_i = None
                        sigma_i = None

                    # ---- COPY GT / ORIGINAL FILE ----
                    orig = os.path.join(dir_path, txt_path)
                    dst_gt = os.path.join(dst, os.path.basename(txt_path))

                    if os.path.exists(orig) and not os.path.exists(dst_gt):
                        copyfile(orig, dst_gt)
                    save_prediction(out_file, transform, kappa_i, sigma_i)

if __name__ == '__main__':
    """
    Example:
      python infer.py --weights mymodel.pth --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/dataset.json
    """
    args = parse_command_line()
    infer(args)
