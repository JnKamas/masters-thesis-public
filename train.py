import os
from torch.utils.data import Subset
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from network import Network
from network_helpers import normalized_l2_loss, parse_command_line, load_model
from dataset import Dataset

def build_rotation_from_yz(y, z, eps=1e-8):
    z = z / (torch.norm(z, dim=1, keepdim=True) + eps)
    y = y - torch.sum(y * z, dim=1, keepdim=True) * z
    y = y / (torch.norm(y, dim=1, keepdim=True) + eps)
    x = torch.cross(y, z, dim=1)
    return torch.stack([x, y, z], dim=2)

def rotation_aleatoric_loss(R_pred, R_gt, s_R, eps=1e-6):
    trace = torch.sum(R_pred.transpose(1,2) * R_gt, dim=(1,2))
    cos = torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)
    theta = torch.acos(cos)

    sigma = torch.nn.functional.softplus(s_R) + eps

    loss = (theta**2) / (sigma**2) + torch.log(sigma**2)
    # loss = theta / sigma + torch.log(sigma)
    return loss.mean()


def get_angles(pred, gt, sym_inv: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates angle between pred and gt vectors.
    Clamping args in acos due to: https://github.com/pytorch/pytorch/issues/8069

    Args:
        pred: (B, 3) tensor of predicted vectors.
        gt:   (B, 3) tensor of ground-truth vectors.
        sym_inv: if True, angle is calculated w.r.t. axis symmetry
                 (e.g., for symmetric bins).
        eps: small constant for numerical stability.

    Returns:
        (B,) tensor of angles in radians.
    """
    pred_norm = torch.norm(pred, dim=-1)
    gt_norm = torch.norm(gt, dim=-1)
    dot = torch.sum(pred * gt, dim=-1)

    if sym_inv:
        cos = torch.clamp(torch.abs(dot / (eps + pred_norm * gt_norm)), -1 + eps, 1 - eps)
    else:
        cos = torch.clamp(dot / (eps + pred_norm * gt_norm), -1 + eps, 1 - eps)

    return torch.acos(cos)


def bayesian_combined_loss(args, preds, targets):
    """
    Combined loss used inside blitz's sample_elbo for Bayesian heads.
    """
    pred_z, pred_y, pred_t = preds
    gt_z, gt_y, gt_t = targets

    loss_z = torch.mean(get_angles(pred_z, gt_z))
    loss_y = torch.mean(get_angles(pred_y, gt_y))
    loss_t = torch.nn.L1Loss()(pred_t, gt_t)

    total_loss = loss_z + loss_y + args.weight * loss_t
    return total_loss, loss_z.detach(), loss_y.detach(), loss_t.detach()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(args)
    model = model.to(device)

    train_dataset = Dataset(
        args.path,
        "train",
        args.input_width,
        args.input_height,
        noise_sigma=args.noise_sigma,
        t_sigma=args.t_sigma,
        random_rot=args.random_rot,
        preload=not args.no_preload,
    )
    if args.modifications == "ensemble" and args.ensemble_type > 1:
        indices = np.random.choice(len(train_dataset), size=len(train_dataset), replace=True)
        boot_dataset = Subset(train_dataset, indices)

        train_loader = DataLoader(
            boot_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

    val_dataset = Dataset(
        args.path,
        "val",
        args.input_width,
        args.input_height,
        preload=not args.no_preload,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_running = 0.0
    loss_rot_running = 0.0
    loss_t_running = 0.0
    loss_z_running = 0.0
    loss_y_running = 0.0

    l1_loss = torch.nn.L1Loss()
    is_bayesian = (args.modifications == "bayesian")

    start_epoch = 0 if args.resume is None else args.resume
    print(f"Starting at epoch {start_epoch}")
    print(f"Running till epoch {args.epochs}")

    train_loss_all = []
    val_loss_all = []

    train_rot_all = []
    train_t_all = []

    for e in range(start_epoch, args.epochs):
        print(f"Starting epoch: {e}")
        model.train()

        epoch_train_loss = []
        epoch_train_rot = []
        epoch_train_t = []

        for sample in train_loader:
            xyz = sample["xyz"].to(device)
            gt_z = sample["bin_transform"][:, :3, 2].to(device)
            gt_y = sample["bin_transform"][:, :3, 1].to(device)
            gt_t = sample["bin_translation"].to(device)

            optimizer.zero_grad()

            # --------------------------------------------------
            # 1. Define loss function (aleatoric vs baseline)
            # --------------------------------------------------

            def compute_loss(pred_z, pred_y, pred_t, s_R, s_t, gt_z, gt_y, gt_t, sample):

                if args.use_aleatoric:
                    # --- ALEATORIC ---
                    R_pred = build_rotation_from_yz(pred_y, pred_z)
                    R_gt = sample["bin_transform"][:, :3, :3].to(device)

                    loss_rot = rotation_aleatoric_loss(R_pred, R_gt, s_R)

                    sigma_t = torch.nn.functional.softplus(s_t) + 1e-6
                    var = sigma_t**2
                    diff = (gt_t - pred_t)

                    loss_t = 0.5 * (torch.log(var) + diff**2 / var)
                    # loss_t = torch.abs(diff) / sigma_t + torch.log(sigma_t)
                    loss_t = args.weight * loss_t.mean()

                    reg_sigma = 0#.01 * sigma_t.mean() # maybe if its too bad.
                    total = loss_rot + loss_t + reg_sigma
                    return total, loss_rot, loss_t, None, None

                else:
                    # --- BASELINE ---
                    loss_z = torch.mean(get_angles(pred_z, gt_z))
                    loss_y = torch.mean(get_angles(pred_y, gt_y))
                    loss_t = args.weight * l1_loss(pred_t, gt_t)

                    total = loss_z + loss_y + loss_t
                    return total, None, loss_t, loss_z, loss_y

            # --------------------------------------------------
            # 2. Training step
            # --------------------------------------------------

            if is_bayesian:

                last_outputs = {}

                def wrapped_loss(preds, targets):
                    if args.use_aleatoric:
                        pred_z, pred_y, pred_t, s_R, s_t = preds
                    else:
                        pred_z, pred_y, pred_t = preds
                        s_R, s_t = None, None

                    gt_z, gt_y, gt_t = targets

                    total, loss_rot, loss_t, loss_z, loss_y = compute_loss(
                        pred_z, pred_y, pred_t, s_R, s_t,
                        gt_z, gt_y, gt_t, sample
                    )

                    last_outputs["loss_rot"] = loss_rot
                    last_outputs["loss_t"] = loss_t
                    last_outputs["loss_z"] = loss_z
                    last_outputs["loss_y"] = loss_y

                    return total

                loss = model.sample_elbo(
                    xyz,
                    (gt_z, gt_y, gt_t),
                    criterion=wrapped_loss,
                    sample_nbr=args.sample_nbr or 3,
                    complexity_cost_weight=args.complexity_cost_weight or 1e-5,
                )

                with torch.no_grad():
                    if args.use_aleatoric:
                        pred_z, pred_y, pred_t, s_R, s_t = model(xyz)
                        data_loss, loss_rot_tmp, loss_t_tmp, loss_z_tmp, loss_y_tmp = compute_loss(
                            pred_z, pred_y, pred_t, s_R, s_t,
                            gt_z, gt_y, gt_t, sample
                        )
                    else:
                        pred_z, pred_y, pred_t = model(xyz)
                        data_loss, loss_rot_tmp, loss_t_tmp, loss_z_tmp, loss_y_tmp = compute_loss(
                            pred_z, pred_y, pred_t, None, None,
                            gt_z, gt_y, gt_t, sample
                        )

                loss_rot = last_outputs.get("loss_rot")
                loss_t   = last_outputs.get("loss_t")
                loss_z   = last_outputs.get("loss_z")
                loss_y   = last_outputs.get("loss_y")

            else:
                # Deterministic / MC-Dropout

                if args.use_aleatoric:
                    pred_z, pred_y, pred_t, s_R, s_t = model(xyz)
                else:
                    pred_z, pred_y, pred_t = model(xyz)
                if args.use_aleatoric:
                    loss, loss_rot, loss_t, loss_z, loss_y = compute_loss(
                        pred_z, pred_y, pred_t, s_R, s_t,
                        gt_z, gt_y, gt_t, sample
                    )
                else:
                    loss, _, loss_t, loss_z, loss_y = compute_loss(
                        pred_z, pred_y, pred_t, None, None,
                        gt_z, gt_y, gt_t, sample
                    )

            rot_loss = (loss_z + loss_y) if loss_z is not None else loss_rot

            epoch_train_loss.append(data_loss.item() if is_bayesian else loss.item())
            epoch_train_rot.append(rot_loss.item())
            epoch_train_t.append((loss_t / args.weight).item())


            # --------------------------------------------------
            # 3. Running loss
            # --------------------------------------------------

            loss_running = 0.9 * loss_running + 0.1 * loss.item()
            if not is_bayesian and args.use_aleatoric:
                sigma_t = torch.nn.functional.softplus(s_t).mean().item()

            PRINT_RUNNING = False
            if PRINT_RUNNING:
                if args.use_aleatoric:
                    print(
                        f"Running loss: {loss_running:.6f}, "
                        f"rot loss: {loss_rot.item() if loss_rot is not None else 0:.6f}, "
                        f"t loss: {loss_t.item() if loss_t is not None else 0:.6f}"
                    )
                else:
                    print(
                        f"Running loss: {loss_running:.6f}, "
                        f"z loss: {loss_z.item() if loss_z is not None else 0:.6f}, "
                        f"y loss: {loss_y.item() if loss_y is not None else 0:.6f}, "
                        f"t loss: {loss_t.item() if loss_t is not None else 0:.6f}"
                    )

            loss.backward()
            optimizer.step()

        train_loss_all.append(np.mean(epoch_train_loss))
        train_rot_all.append(np.mean(epoch_train_rot))
        train_t_all.append(np.mean(epoch_train_t))

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_losses = []          # total objective
            val_losses_rot = []      # rotation
            val_losses_t = []        # translation
            val_angle_z = []         # metrics only
            val_angle_y = []

            for sample in val_loader:
                xyz = sample["xyz"].to(device)
                R_gt = sample["bin_transform"][:, :3, :3].to(device)
                gt_z = R_gt[:, :, 2]
                gt_y = R_gt[:, :, 1]
                gt_t = sample["bin_translation"].to(device)
                
                # PREDICTIONS
                if is_bayesian:
                    preds = [model(xyz) for _ in range(3)]

                    loss_samples = []
                    loss_rot_samples = []
                    loss_t_samples = []

                    for p in preds:
                        if args.use_aleatoric:
                            pred_z_i, pred_y_i, pred_t_i, s_R_i, s_t_i = p
                            R_pred_i = build_rotation_from_yz(pred_y_i, pred_z_i)

                            loss_rot_i = rotation_aleatoric_loss(R_pred_i, R_gt, s_R_i)

                            sigma_t_i = torch.nn.functional.softplus(s_t_i) + 1e-6
                            var_i = sigma_t_i**2
                            diff_i = (gt_t - pred_t_i)

                            loss_t_i = 0.5 * (torch.log(var_i) + diff_i**2 / var_i)
                            loss_t_i = args.weight * loss_t_i.mean()

                            loss_i = loss_rot_i + loss_t_i

                        else:
                            pred_z_i, pred_y_i, pred_t_i = p
                            loss_rot_i = torch.mean(get_angles(pred_z_i, gt_z)) + \
                                        torch.mean(get_angles(pred_y_i, gt_y))

                            loss_t_i = args.weight * l1_loss(pred_t_i, gt_t)

                            loss_i = loss_rot_i + loss_t_i

                        loss_samples.append(loss_i)
                        loss_rot_samples.append(loss_rot_i)
                        loss_t_samples.append(loss_t_i)

                    loss = torch.mean(torch.stack(loss_samples))
                    loss_rot = torch.mean(torch.stack(loss_rot_samples))
                    loss_t = torch.mean(torch.stack(loss_t_samples))

                    # for metrics (still OK to use mean prediction)
                    pred_z = torch.mean(torch.stack([p[0] for p in preds]), dim=0)
                    pred_y = torch.mean(torch.stack([p[1] for p in preds]), dim=0)
                    pred_t = torch.mean(torch.stack([p[2] for p in preds]), dim=0)

                    if args.use_aleatoric:
                        s_R_samples = torch.stack([p[3] for p in preds])
                        s_t_samples = torch.stack([p[4] for p in preds])

                        sigma_t = torch.mean(torch.nn.functional.softplus(s_t_samples), dim=0)

                        s_R = s_R_samples.mean(dim=0)


                else:
                    if args.use_aleatoric:
                        pred_z, pred_y, pred_t, s_R, s_t = model(xyz)
                        sigma_t = torch.nn.functional.softplus(s_t) + 1e-6
                    else:
                        pred_z, pred_y, pred_t = model(xyz)

                # LOSSES
                if not is_bayesian:
                    if args.use_aleatoric:
                        R_pred = build_rotation_from_yz(pred_y, pred_z)
                        loss_rot = rotation_aleatoric_loss(R_pred, R_gt, s_R)

                        var = sigma_t**2
                        diff = (gt_t - pred_t)
                        loss_t = 0.5 * (torch.log(var) + diff**2 / var)
                        # loss_t = torch.abs(diff) / sigma_t + torch.log(sigma_t)
                        loss_t = args.weight * loss_t.mean()

                        loss = loss_rot + loss_t

                    else:
                        loss_rot = torch.mean(get_angles(pred_z, gt_z)) + \
                                torch.mean(get_angles(pred_y, gt_y, sym_inv=True))

                        loss_t = args.weight * l1_loss(pred_t, gt_t)

                        loss = loss_rot + loss_t

                # METRICS
                angle_z = torch.mean(get_angles(pred_z, gt_z))
                angle_y = torch.mean(get_angles(pred_y, gt_y, sym_inv=True))

                val_losses.append(loss.item())
                val_losses_rot.append(loss_rot.item())
                val_losses_t.append((loss_t / args.weight).item())
                val_angle_z.append(angle_z.item())
                val_angle_y.append(angle_y.item())

            print(20 * "*")
            print(f"Epoch {e}/{args.epochs}")

            print(
                f"TRAIN - loss: {train_loss_all[-1]:.6f}, "
                f"rot: {train_rot_all[-1]:.6f}, "
                f"t: {train_t_all[-1]:.6f}"
            )

            print(
                f"VAL   - loss: {np.mean(val_losses):.6f}, "
                f"rot: {np.mean(val_losses_rot):.6f}, "
                f"t: {np.mean(val_losses_t):.6f}"
            )

            val_loss_all.append(np.mean(val_losses))


        # ---------------------------------------------------------------------
        # Checkpoints & logging
        # ---------------------------------------------------------------------
        if True: #args.dump_every != 0 and e % args.dump_every == 0:
            print("Saving checkpoint")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{e:03d}.pth")

        with open("loss_log.txt", "a") as f:
            f.write(f"{e+1}\t{loss_running:.6f}\t{np.mean(val_losses):.6f}\n")

    # Final model save
    os.makedirs("models", exist_ok=True)
    final_model_name = f"models/{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(model.state_dict(), final_model_name)

    # Save loss curves
    np.set_printoptions(suppress=True)
    np.savetxt("train_err.out", np.array(train_loss_all), delimiter=",")
    np.savetxt("val_err.out", np.array(val_loss_all), delimiter=",")
    np.savetxt("train_rot.out", np.array(train_rot_all), delimiter=",")
    np.savetxt("train_t.out", np.array(train_t_all), delimiter=",")

    # Save losses plot
    epochs = np.arange(len(train_loss_all))

    plt.figure()
    plt.plot(epochs, train_loss_all, label="train")
    plt.plot(epochs, val_loss_all, label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    plt.savefig("loss_curve.png")
    plt.close()

if __name__ == "__main__":
    """
    Example:
        python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/dataset.json

    For MC-dropout with backbone dropout, something like:
        --modifications mc_dropout --dropout_prob 0.1 --dropout_prob_backbone 0.1
    (depending on how parse_command_line defines these flags)
    """
    args = parse_command_line()
    train(args)
    