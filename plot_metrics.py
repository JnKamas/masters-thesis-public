import matplotlib.pyplot as plt
import numpy as np  

def plot_rotation_metrics(spread_list, entropy_list, orth_dev_list, det_dev_list,
                          angles_samples=None, angles_between=None,
                          save_path="uncertainty_rotation.png"):
    num = 5
    fig, ax = plt.subplots(num, 1, figsize=(10, 4 * num))
    idx = 0

    # 1) sample spread scatter + mean
    if angles_between is not None:
        means = []
        for si, disp in enumerate(angles_between):
            arr = np.full_like(disp, si)
            ax[idx].scatter(arr, disp, color='red', alpha=0.1, s=10,
                            label='Pairwise Angles' if si == 0 else "")
            means.append(np.mean(disp))
        ax[idx].scatter(range(len(means)), means, color='darkred', marker='x', s=50, label='Mean Pairwise Angle')
    ax[idx].plot(range(len(spread_list)), spread_list, 'g.-', label='Sample Spread')
    ax[idx].set_title('Rotation Spread per Sample')
    ax[idx].set_ylabel('Angle (rad)')
    ax[idx].set_ylim(0, np.pi)
    ax[idx].legend()
    idx += 1

    # 2) entropy
    ax[idx].plot(range(len(entropy_list)), entropy_list, 'm.-', label='Entropy')
    ax[idx].set_title('Rotation Entropy-like Measure')
    ax[idx].set_ylabel('Entropy (nats)')
    ax[idx].legend()
    idx += 1

    # 3) orthogonality deviation
    ax[idx].plot(range(len(orth_dev_list)), orth_dev_list, 'c.-', label='Δ_orth')
    ax[idx].set_title('Ortogonalitná odchýlka (Δ_orth)')
    ax[idx].set_ylabel('‖RᵀR - I‖_F')
    ax[idx].legend()
    idx += 1

    # 4) determinant deviation
    ax[idx].plot(range(len(det_dev_list)), det_dev_list, 'y.-', label='Δ_det')
    ax[idx].set_title('Determinantná odchýlka (Δ_det)')
    ax[idx].set_ylabel('|det(R) - 1|')
    ax[idx].legend()
    idx += 1

    # 5) angular error to GT
    if angles_samples is not None:
        arr = np.array(angles_samples).T
        mean_ang = arr.mean(axis=0)
        low = np.percentile(arr, 2.5, axis=0)
        high = np.percentile(arr, 97.5, axis=0)
        for j in range(arr.shape[0]):
            ax[idx].scatter(range(arr.shape[1]), arr[j], color='blue', alpha=0.1, s=10)
        ax[idx].plot(range(len(mean_ang)), mean_ang, 'b-', label='Mean Angular Error')
        ax[idx].fill_between(range(len(mean_ang)), low, high, color='blue', alpha=0.3, label='95% CI')
        ax[idx].set_title('Rotation Angular Errors to GT')
        ax[idx].set_ylabel('Error (rad)')
        ax[idx].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_translation_uncertainty(pred_samples, gt_values, save_path="uncertainty_translation.png"):
    pred = np.array(pred_samples)
    # gt_values may be list of tensors or ndarrays
    gt_list = []
    for gt in gt_values:
        if hasattr(gt, 'cpu'):
            gt_list.append(gt.cpu().numpy())
        else:
            gt_list.append(gt)
    gt_arr = np.array(gt_list)

    means = pred.mean(axis=0)
    lower = np.percentile(pred, 2.5, axis=0)
    upper = np.percentile(pred, 97.5, axis=0)

    comps = ['t[0]', 't[1]', 't[2]']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        for s in range(pred.shape[0]):
            ax[i].scatter(range(pred.shape[1]), pred[s, :, i], color='blue', alpha=0.1, s=10)
        ax[i].plot(means[:, i], 'b-', label='Mean')
        ax[i].fill_between(range(means.shape[0]), lower[:, i], upper[:, i], color='blue', alpha=0.3)
        ax[i].scatter(range(gt_arr.shape[0]), gt_arr[:, i], color='red', marker='x', s=50, label='GT')
        ax[i].set_title(f'Uncertainty in {comps[i]}')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
