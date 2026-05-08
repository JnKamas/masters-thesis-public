# Info

This folder is **based on** the original repo [bin-detect](https://github.com/gajdosech2/bin-detect).  
**Note:** This is not an official GitHub fork, but rather a clone with my own modifications.

## Overview

This is the repository for my masters thesis. The repository contains code for training and evaluating neural networks that estimate 6D bin pose from EXR point-cloud images. It extends the original `bin-detect` project with uncertainty-aware variants, including MC Dropout, Bayesian heads, ensembles, and aleatoric uncertainty outputs.

Main entry points:

- `train.py` - train a model from a dataset JSON.
- `infer.py` - export predicted transforms for the validation/test split.
- `evaluate.py` - compute translation, rotation, and uncertainty metrics from predictions.
- `run_model.py` - run inference and evaluation for a stored model in one command.

## Setup

Create the Conda environment:

```bash
conda env create -f env.yml
conda activate kamas
```

The environment expects CUDA-capable PyTorch. If your CUDA version differs from the one in `env.yml`, install the matching PyTorch build before running training or inference.

## Data

The dataset is provided as a JSON file whose entries reference EXR position images and transform text files. Paths inside the JSON are resolved relative to the JSON file location.

Typical expected files:

- `dataset.json` - list of samples.
- EXR position maps referenced by `exr_positions_path`.
- Ground-truth transform text files referenced by each sample entry.

If the JSON path does not include `train`, `val`, or `test`, the loader creates a deterministic split: every fifth sample is used for validation and the remaining samples for training.

## Training

Run baseline training:

```bash
python train.py /path/to/dataset.json
```

Common options:

```bash
python train.py /path/to/dataset.json \
  --batch_size 8 \
  --epochs 250 \
  --backbone resnet34 \
  --gpu 0
```

Useful variants:

```bash
# MC Dropout
python train.py /path/to/dataset.json -mod mc_dropout -dpt 0.1 -dpr 0.1 -dpb 0.1

# Bayesian heads
python train.py /path/to/dataset.json -mod bayesian -sn 3 -ccw 0.001 -bt 0

# Aleatoric uncertainty
python train.py /path/to/dataset.json --use_aleatoric
```

Checkpoints are written to `checkpoints/` by the training script.

## Inference

Run inference with a checkpoint:

```bash
python infer.py /path/to/dataset.json --weights_path /path/to/model.pth --out_dir inference/model_name
```

For stochastic methods, set the modification and sample count:

```bash
python infer.py /path/to/dataset.json \
  --weights_path /path/to/model.pth \
  --out_dir inference/model_name \
  -mod mc_dropout \
  --mc_samples 50
```

Prediction files are written as transform `.txt` files. When aleatoric uncertainty is enabled, prediction files also include `kappa` and translation sigma values.

## Evaluation

Evaluate an inference output directory:

```bash
python evaluate.py inference/model_name
```

For stochastic predictions:

```bash
python evaluate.py inference/model_name -mod mc_dropout --mc_samples 50
```

The evaluator reports pose errors and uncertainty statistics from the generated prediction files.

## Combined Run

Use `run_model.py` to run inference and evaluation for a model stored under `models/`:

```bash
python run_model.py model_name --dataset /path/to/dataset.json
```

Example for MC Dropout:

```bash
python run_model.py model_name \
  --dataset /path/to/dataset.json \
  -mod mc_dropout \
  -mc 50 \
  -dpt 0.1 \
  -dpr 0.1 \
  -dpb 0.1
```


---

# Sources for the data

## Gajdosech et al. (2021) - [bin-detect](https://github.com/gajdosech2/bin-detect)

**Weights:**  
https://liveuniba-my.sharepoint.com/:f:/g/personal/gajdosech2_uniba_sk/EmTGItM-mMpLmDfvOCRhwKABu2KSYfouVyeh_L8UPNonCA?e=DjOGfq

**Dataset:**  
https://liveuniba-my.sharepoint.com/:f:/g/personal/gajdosech2_uniba_sk/EtTuc2-ccudNkgQSqKiNFqUBBZw2WQNWyRs2Th4yXOoODQ?e=i5twkn

**Dataset via Seletex link:**
https://liveuniba-my.sharepoint.com/personal/madaras2_uniba_sk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmadaras2%5Funiba%5Fsk%2FDocuments%2FGajdosech%5Fetal%5F2021%5Fdataset&ga=1

## Mok et al. (2026) - [LETR3D](https://github.com/4zzz/LETR3D)

**Data**:
https://415102.xyz/share/8qK7ZAuRs