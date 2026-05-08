#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import time

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser(proj_root):
    parser = argparse.ArgumentParser(
        description="Run inference + evaluation in one command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "model_name",
        help="Name of the model (without .pth), e.g. bayes1300"
    )

    # Model / backbone
    parser.add_argument("-mod", "--modifications",
                        choices=["mc_dropout", "bayesian", "ensemble"],
                        default=None,
                        help="Modification type (options: mc_dropout, bayesian, ensemble)")
    parser.add_argument("-bb", "--backbone",
                        default="resnet34",
                        help="Backbone for inference")

    # Inference settings
    parser.add_argument("-mc", "--mc_samples", type=int, default=50,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--no_preload", action="store_true",
                        help="Pass --no_preload to infer.py")
    parser.add_argument("-iw", "--input_width", type=int, default=516,
                        help="Input width for inference")
    parser.add_argument("-ih", "--input_height", type=int, default=386,
                        help="Input height for inference")

    # Paths
    parser.add_argument("--dataset",
                        default=os.path.expanduser("~/thesis/large-data/test/dataset.json"),
                        help="Path to dataset JSON")
    parser.add_argument("--models_dir",
                        default=os.path.join(proj_root, "models"),
                        help="Directory containing .pth files")
    parser.add_argument("--inference_dir",
                        default=os.path.join(proj_root, "inference"),
                        help="Base output dir")

    # Dropout / UQ
    parser.add_argument("-dpt", "--dropout_prob_trans", type=float, default=0.0,
                        help="Dropout probability for translation")
    parser.add_argument("-dpr", "--dropout_prob_rot", type=float, default=0.0,
                        help="Dropout probability for rotation")
    parser.add_argument("-dpb", "--dropout_prob_backbone", type=float, default=0.0,
                        help="Dropout probability for ResNet backbone residual blocks")

    parser.add_argument("-bs", "--bootstrap_samples", type=int, default=0,
                        help="Number of bootstrapped ensemble models, 0 to disable")
    parser.add_argument("-et", "--ensemble_type", type=int, default=1,
                        help="Ensemble type")

    parser.add_argument("-sn", "--sample_nbr", type=int, default=200,
                        help="Sample number for MC Dropout")
    parser.add_argument("-ccw", "--complexity_cost_weight", type=float, default=0.001,
                        help="Weight for complexity cost in Bayesian layers")
    parser.add_argument("-bt", "--bayesian_type", type=int, default=0,
                        help="Bayesian type")
    parser.add_argument("-is", "--input_sigma", type=float, default=0.1,
                        help="Input sigma for Bayesian layers")
    parser.add_argument("-ale", "--use_aleatoric", action="store_true",
                        help="Use aleatoric uncertainty")


    parser.add_argument("--eval_only", type=bool, default=False,
                        help="If True, only runs evaluation on existing inference results")
    return parser


# ------------------------------------------------------------
# Command Builders
# ------------------------------------------------------------
def build_infer_cmd(args, infer_script, weights_path):
    cmd = [
        sys.executable, infer_script,
        "-bb", args.backbone,
        "-iw", str(args.input_width),
        "-ih", str(args.input_height),
        "-dpt", str(args.dropout_prob_trans),
        "-dpr", str(args.dropout_prob_rot),
        "-dpb", str(args.dropout_prob_backbone),
        "-b", str(args.batch_size),
        "-sn", str(args.sample_nbr),
        "-ccw", str(args.complexity_cost_weight),
        "-bt", str(args.bayesian_type),
        "-is", str(args.input_sigma),
        "-et", str(args.ensemble_type),
        *(["--use_aleatoric"] if args.use_aleatoric else []),
        "--weights_path", weights_path,
        "--mc_samples", str(args.mc_samples),
        "--bootstrap_samples", str(args.bootstrap_samples),
        args.dataset,
    ]

    if args.modifications:
        cmd += ["-mod", args.modifications]

    if args.no_preload:
        cmd.append("--no_preload")

    return cmd


def build_eval_cmd(args, eval_script, inference_output_dir):
    mc_samples = args.mc_samples

    # For ensemble, mc_samples = number of ensemble members
    if args.modifications == "ensemble":
        models_dir = os.path.join(args.models_dir, args.model_name)
        mc_samples = len([
            f for f in os.listdir(models_dir)
            if f.endswith(".pth")
        ])

    cmd = [ 
        sys.executable, eval_script,
        "--mc_samples", str(mc_samples),
        "--bootstrap_samples", str(args.bootstrap_samples),
        inference_output_dir
    ]

    if args.modifications:
        cmd += ["--modifications", args.modifications]

    if args.use_aleatoric:
        cmd += ["--use_aleatoric"]

    return cmd



# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
def save_results(proj_root, args, evaluated_block):
    results_dir = os.path.join(proj_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    name = args.model_name
    name += f"_dpt{args.dropout_prob_trans}"
    name += f"_dpr{args.dropout_prob_rot}"
    name += f"_dp{args.dropout_prob}"
    if args.modifications:
        name += f"_{args.modifications}"

    result_file = os.path.join(results_dir, name + ".txt")

    with open(result_file, "w") as f:
        for line in evaluated_block:
            f.write(line + "\n")

    return result_file

# ------------------------------------------------------------
# Merge ensembles into MC-like structure
# ------------------------------------------------------------
def merge_ensemble_as_mc(inference_dir):
    """
    Convert:
      inference/run/ensXX/dataset/prediction_scan_YYY.txt
    into:
      inference/run_merged/dataset/prediction{i}_scan_YYY.txt
    """
    merged_dir = inference_dir + "_merged"

    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)

    members = sorted(
        d for d in os.listdir(inference_dir)
        if os.path.isdir(os.path.join(inference_dir, d))
    )

    datasets = set()
    for m in members:
        mdir = os.path.join(inference_dir, m)
        for d in os.listdir(mdir):
            if os.path.isdir(os.path.join(mdir, d)):
                datasets.add(d)

    for dataset in datasets:
        out_ds = os.path.join(merged_dir, dataset)
        os.makedirs(out_ds, exist_ok=True)

        # copy GT once (from first member)
        first_member_ds = os.path.join(inference_dir, members[0], dataset)
        for f in os.listdir(first_member_ds):
            if f.startswith("scan_") and f.endswith(".txt"):
                shutil.copyfile(
                    os.path.join(first_member_ds, f),
                    os.path.join(out_ds, f),
                )

        # copy predictions as MC samples
        for mi, member in enumerate(members):
            src_ds = os.path.join(inference_dir, member, dataset)
            for f in os.listdir(src_ds):
                if f.startswith("prediction_scan_"):
                    scan_id = f.replace("prediction_scan_", "")
                    dst = f"prediction{mi}_scan_{scan_id}"
                    shutil.copyfile(
                        os.path.join(src_ds, f),
                        os.path.join(out_ds, dst),
                    )

    return merged_dir

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    proj_root = os.path.normpath(os.path.join(script_dir, ".."))

    parser = build_parser(proj_root)
    args = parser.parse_args()

    infer_script = os.path.join(script_dir, "infer.py")
    eval_script = os.path.join(script_dir, "evaluate.py")

    inference_output_dir = os.path.join(args.inference_dir, args.model_name)

    # Clean old infer output
    if not args.eval_only:
        if os.path.exists(inference_output_dir):
            shutil.rmtree(inference_output_dir)
        os.makedirs(inference_output_dir, exist_ok=True)

    # -----------------------------
    # 1) INFERENCE
    # -----------------------------
    start_time = time.time()
    if not args.eval_only:

        if args.modifications == "ensemble":

            ensemble_models_dir = os.path.join(args.models_dir, args.model_name)
            ensemble_out_dir = os.path.join(args.inference_dir, args.model_name)

            if not os.path.isdir(ensemble_models_dir):
                raise FileNotFoundError(f"Ensemble folder not found: {ensemble_models_dir}")

            ckpts = sorted(f for f in os.listdir(ensemble_models_dir) if f.endswith(".pth"))

            # ensemble bootstrap
            if args.bootstrap_samples > 0:
                B = args.bootstrap_samples
                M = len(ckpts)

                idx = np.random.choice(M, B, replace=True)
                ckpts = [ckpts[i] for i in idx]

            if not ckpts:
                raise RuntimeError(f"No .pth files in ensemble folder {ensemble_models_dir}")

            os.makedirs(ensemble_out_dir, exist_ok=True)

            for mi, ckpt in enumerate(ckpts):
                member_name = f"{os.path.splitext(ckpt)[0]}_{mi}"
                member_out_dir = os.path.join(ensemble_out_dir, member_name)
                os.makedirs(member_out_dir, exist_ok=True)

                weights_path = os.path.join(ensemble_models_dir, ckpt)

                infer_cmd = build_infer_cmd(args, infer_script, weights_path)

                # IMPORTANT: force infer.py output directory
                infer_cmd += ["--out_dir", member_out_dir]

                print("▶︎ Inference (ensemble):", " ".join(infer_cmd))
                if subprocess.run(infer_cmd).returncode != 0:
                    sys.exit(1)

        else:
            weights_path = os.path.join(args.models_dir, args.model_name + ".pth")
            infer_cmd = build_infer_cmd(args, infer_script, weights_path)

            print("▶︎ Inference:", " ".join(infer_cmd))
            if subprocess.run(infer_cmd).returncode != 0:
                sys.exit(1)
    end_time = time.time()
    # -----------------------------
    # 2) EVALUATION
    # -----------------------------
    eval_input_dir = inference_output_dir

    if args.modifications == "ensemble":
        eval_input_dir = merge_ensemble_as_mc(inference_output_dir)

    eval_cmd = build_eval_cmd(args, eval_script, eval_input_dir)

    print("▶︎ Evaluation:", " ".join(eval_cmd))

    result = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(result.stdout)
    print(f"✔ Inference completed in {end_time - start_time:.2f} s")
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
