# UAMIPL: Uncertainty-Aware Multi-Instance Partial-Label Learning via Evidential Deep Model

This repository provides the official PyTorch implementation of our paper:

**Uncertainty-Aware Multi-Instance Partial-Label Learning via Evidential Deep Model**

## Overview

Multi-Instance Partial-Label Learning (MIPL) is a weakly supervised learning setting in which each example is represented as a bag of instances and is associated with a candidate label set containing the latent ground-truth label. This repository implements **UAMIPL**, which introduces uncertainty-aware instance aggregation by combining:

- **Bayesian global uncertainty modeling** over the instance-scoring parameters
- **Local evidential uncertainty estimation** at the instance level
- **Reliability-aware aggregation** for robust bag representation learning
- **Candidate-label refinement** for MIPL disambiguation

The repository also includes the experimental settings used in the paper, together with additional options for:

- **uniform candidate-label weighting**
- **reliability shuffling sanity-check**

## Environment

The dependency configuration for this project is provided in `environment.yml`.

To create the conda environment, run:

```sh
conda env create -f environment.yml -n UAMIPL
```

Then activate it via:

```sh
conda activate UAMIPL
```

## Datasets

The benchmark MIPL datasets used in this work are publicly available and can be obtained from:

http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data

Please place the downloaded datasets under the expected data directory before training.

The real-world CRC-MIPL datasets follow the same benchmark setting used in prior MIPL studies.

## Running the Code

### 1. Benchmark Experiments

Example commands for benchmark datasets:

```sh
python main.py --ds FMNIST_MIPL --ds_suffix r1 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 30 --lambda_edl 0.2
python main.py --ds FMNIST_MIPL --ds_suffix r2 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 40 --lambda_edl 0.2
python main.py --ds FMNIST_MIPL --ds_suffix r3 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 20 --lambda_edl 0.2
```

You may similarly replace `FMNIST_MIPL` with other datasets such as:

- `MNIST_MIPL`
- `Birdsong_MIPL`
- `SIVAL_MIPL`
- CRC-MIPL variants used in the paper

### 2. Uniform Candidate-Label Weighting Baseline

To run the static uniform candidate-label weighting variant, add:

```sh
--uniform_candidate_weighting
```

Example:

```sh
python main.py --ds FMNIST_MIPL --ds_suffix r2 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 40 --lambda_edl 0.2 --uniform_candidate_weighting
```

### 3. Reliability-Shuffling Sanity-Check

To run the shuffled-reliability sanity-check experiment, add:

```sh
--shuffle_reliability
```

Example:

```sh
python main.py --ds FMNIST_MIPL --ds_suffix r2 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 40 --lambda_edl 0.2 --shuffle_reliability
```

## Main Arguments

Some important arguments are listed below:

- `--ds`: dataset name
- `--ds_suffix`: ambiguity setting for benchmark datasets, e.g. `r1`, `r2`, `r3`
- `--bs`: batch size
- `--lr`: learning rate
- `--epoch`: number of training epochs
- `--nr_samples`: number of posterior samples
- `--lambda_edl`: uncertainty calibration coefficient
- `--uniform_candidate_weighting`: use static uniform candidate-label weighting instead of dynamic candidate-label refinement
- `--shuffle_reliability`: shuffle the reliability signal for sanity-check analysis

## Parameter Settings

The parameter settings used in our experiments are summarized below.

| Dataset | Learning Rate | Batch Size | Epochs | `nr_samples` | `lambda_edl` |
|--------|---------------|------------|--------|--------------|--------------|
| MNIST_MIPL (`r=1`) | 5e-4 | 350 | 200 | 40 | 0.01 |
| MNIST_MIPL (`r=2`) | 5e-4 | 350 | 200 | 30 | 0.01 |
| MNIST_MIPL (`r=3`) | 5e-4 | 350 | 200 | 50 | 0.1 |
| FMNIST_MIPL (`r=1`) | 5e-4 | 350 | 200 | 30 | 0.01 |
| FMNIST_MIPL (`r=2`) | 5e-4 | 350 | 200 | 40 | 0.05 |
| FMNIST_MIPL (`r=3`) | 5e-4 | 350 | 200 | 20 | 0.05 |
| Birdsong_MIPL (`r=1`) | 0.001 | 910 | 500 | 10 | 0.2 |
| Birdsong_MIPL (`r=2`) | 0.002 | 910 | 500 | 20 | 0.1 |
| Birdsong_MIPL (`r=3`) | 0.005 | 910 | 500 | 30 | 0.5 |
| SIVAL_MIPL (`r=1`) | 0.002 | 1050 | 500 | 30 | 0.05 |
| SIVAL_MIPL (`r=2`) | 0.005 | 1050 | 500 | 20 | 0.05 |
| SIVAL_MIPL (`r=3`) | 0.005 | 1050 | 500 | 30 | 0.2 |
| CRC-MIPL-Row | 0.001 | 4900 | 500 | 10 | 0.1 |
| CRC-MIPL-SBN | 0.001 | 4900 | 500 | 10 | 0.1 |
| CRC-MIPL-KMeansSeg | 0.001 | 4900 | 500 | 10 | 0.1 |
| CRC-MIPL-SIFT | 0.001 | 4900 | 500 | 10 | 0.1 |
