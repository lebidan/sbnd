# Experiments

This page cross-lists every experiment config under [`conf/exp/`](../conf/exp) with the matching evaluation logs in [`log/test/`](../log/test). Use it to find, for a given code, which model and training setup performs best — eval logs are CSV files with WER/BER per Eb/N₀ point that can be opened directly in any spreadsheet or plotting tool.

These are just reasonable baselines. Help us improve on them either by closing the gap to MLD or by finding ways to use fewer trainable parameters at the same performance level.

Some of the eval logs below have been used to generate the plots in [this preprint](https://arxiv.org/abs/2605.03620).

**See also:** [README](../README.md) · [Training a model](training.md) · [Evaluating a model](evaluation.md) · [Extending SBND](extending.md)

## Contents

1. [BCH (31, 21, 5)](#bch-31-21-5)
2. [BCH (63, 45, 7)](#bch-63-45-7)
3. [eBCH (32, 16, 8)](#ebch-32-16-8)
4. [Reed-Muller (32, 16, 8)](#reed-muller-32-16-8)
5. [QC-LDPC (96, 48, 10) from RPTU](#qc-ldpc-96-48-10-from-rptu)
6. [Polar (128, 64, 8) from RPTU](#polar-128-64-8-from-rptu)

Conventions:

- The **Decoder** column links to the experiment config file. The **Size** column gives the number of trainable parameters. The **Eval log** column links to the CSV with `sbnd-test` results; *epochs* is the number of training epochs of the evaluated checkpoint.
- The **Training data** column summarizes the training-time data pipeline: `on-demand at X dB` for samples generated on the fly at a fixed Eb/N₀, or `<size> ML samples at X dB` for a pre-computed ML-error-pattern dataset (see [`docs/training.md`](training.md#training-data)). The mention `with augmentation` indicates that code-automorphism data augmentation is enabled.
- Test-time scaling (TTS) variants of an eval log are listed alongside the base log when available — see [Test-time scaling](evaluation.md#3-test-time-scaling).

## BCH (31, 21, 5)

Code file: [`bch.31.21.mat`](../data/codes/bch.31.21.mat). All experiments use the 4M-sample ML dataset [`train-ml-4M-3dB.mat`](https://sdrive.cnrs.fr/s/bKBHagxAwLiNNzn) (see [Pre-computed dataset format and download](training.md#pre-computed-dataset-format-and-download)) or its 16M / permuted variants. The 16M dataset can be download from [this link](https://ai4code.projects.labsticc.fr/software/).

### Experiments with a pre-computed dataset

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [GRU](../conf/exp/gru-bch-31-21-ml-1m-3dB-aug.yaml) | 632K | 1M ML samples at 3 dB, with augmentation | 512 | [results](../log/test/gru-bch-31-21-ml-1m-3dB-aug-512epochs-zesty-capybara-1841.csv) |
| [GRU](../conf/exp/gru-bch-31-21-ml-4m-3dB.yaml) | 632K | 4M ML samples at 3 dB | 128 | [results](../log/test/gru-bch-31-21-ml-4m-3dB-128epochs-stellar-paper-1839.csv) |
| [GRU](../conf/exp/gru-bch-31-21-ml-4m-3dB-aug.yaml) | 632K | 4M ML samples at 3 dB, with augmentation | 128 | [results](../log/test/gru-bch-31-21-ml-4m-3dB-aug-128epochs-dainty-snowflake-2008.csv) |
| [GRU](../conf/exp/gru-bch-31-21-ml-16m-3dB.yaml) | 632K | 16M ML samples at 3 dB | 128 | [results](../log/test/gru-bch-31-21-ml-16m-3dB-128epochs-wandering-energy-1840.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-1m-3dB.yaml) | 1.2M | 1M ML samples at 3 dB | 128 | [results](../log/test/ecct-bch-31-21-ml-1m-3dB-128epochs-celestial-fog-1911.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-1m-3dB-aug.yaml) | 1.2M | 1M ML samples at 3 dB, with augmentation | 512 | [results](../log/test/ecct-bch-31-21-ml-1m-3dB-aug-512epochs-resilient-sun-1793.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-1mx4perm-3dB.yaml) | 1.2M | 1M ML samples × 4 permutations at 3 dB | 128 | [results](../log/test/ecct-bch-31-21-ml-1mx4perm-3dB-128epochs-worthy-feather-1910.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-1mx16perm-3dB.yaml) | 1.2M | 1M ML samples × 16 permutations at 3 dB | 128 | [results](../log/test/ecct-bch-31-21-ml-1mx16perm-3dB-128epochs-happy-night-1912.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-4m-3dB.yaml) | 1.2M | 4M ML samples at 3 dB | 128 | [results](../log/test/ecct-bch-31-21-ml-4m-3dB-128epochs-woven-cherry-1909.csv) |
| [ECCT](../conf/exp/ecct-bch-31-21-ml-16m-3dB.yaml) | 1.2M | 16M ML samples at 3 dB | 128 | [results](../log/test/ecct-bch-31-21-ml-16m-3dB-128epochs-daily-haze-1907.csv) |
| [CrossMPT](../conf/exp/crossmpt-bch-31-21-ml-1m-3dB-aug.yaml) | 1.2M | 1M ML samples at 3 dB, with augmentation | 512 | [results](../log/test/crossmpt-bch-31-21-ml-1m-3dB-aug-512epochs-frosty-pyramid-1843.csv) |
| [rECCT](../conf/exp/recct-bch-31-21-ml-1m-3dB-aug.yaml) | 70K | 1M ML samples at 3 dB, with augmentation | 512 | [results](../log/test/recct-bch-31-21-ml-1m-3dB-aug-512epochs-stilted-wood-1858.csv) |
| [rECCT](../conf/exp/recct-bch-31-21-ml-4m-3dB.yaml) | 70K | 4M ML samples at 3 dB | 128 | [results](../log/test/recct-bch-31-21-ml-4m-3dB-128epochs-vibrant-paper-1856.csv) |
| [rECCT](../conf/exp/recct-bch-31-21-ml-16m-3dB.yaml) | 70K | 16M ML samples at 3 dB | 128 | [results](../log/test/recct-bch-31-21-ml-16m-3dB-128epochs-iconic-bee-2007.csv) |

## BCH (63, 45, 7)

Code file: [`bch.63.45.mat`](../data/codes/bch.63.45.mat). Dataset for the dataset-based experiments below: [`train-ml-4M-2dB.mat`](https://sdrive.cnrs.fr/s/wMDN6beY2Gnb7rg) (see [Pre-computed dataset format and download](training.md#pre-computed-dataset-format-and-download)). Larger ML datasets can also be downloaded from [this link](https://ai4code.projects.labsticc.fr/software/).

### Experiments with on-demand data

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [ECCT](../conf/exp/ecct-bch-63-45-on-demand-2dB.yaml) | 1.2M | on-demand at 2 dB | 128 | [results](../log/test/ecct-bch-63-45-on-demand-2dB-128epochs-faithful-frost-1882.csv) |
| [ECCT](../conf/exp/ecct-bch-63-45-on-demand-2dB.yaml) | 1.2M | on-demand at 2 dB | 512 | [results](../log/test/ecct-bch-63-45-on-demand-2dB-512epochs-comic-durian-1881.csv) |
| [CrossMPT](../conf/exp/crossmpt-bch-63-45-on-demand-3dB.yaml) | 1.2M | on-demand at 3 dB | 128 | [results](../log/test/crossmpt-bch-63-45-on-demand-3dB-128epochs-bright-glade-2001.csv) |
| [CrossMPT](../conf/exp/crossmpt-bch-63-45-on-demand-3dB.yaml) | 1.2M | on-demand at 3 dB | 512 | [results](../log/test/crossmpt-bch-63-45-on-demand-3dB-512epochs-ethereal-plasma-2006.csv) |

### Experiments with a pre-computed dataset

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [GRU](../conf/exp/gru-bch-63-45-ml-4m-2dB-aug.yaml) | 1.4M | 4M ML samples at 2 dB, with augmentation | 512 | [results](../log/test/gru-bch-63-45-ml-4m-2dB-aug-512epochs-smart-star-1842.csv) |
| [ECCT](../conf/exp/ecct-bch-63-45-ml-1m-2dB-aug.yaml) | 1.2M | 1M ML samples at 2 dB, with augmentation | 128 | [results](../log/test/ecct-bch-63-45-ml-1m-2dB-aug-128epochs-restful-silence-1903.csv) |
| [ECCT](../conf/exp/ecct-bch-63-45-ml-4m-2dB-aug.yaml) | 1.2M | 4M ML samples at 2 dB, with augmentation | 512 | [results](../log/test/ecct-bch-63-45-ml-4m-2dB-aug-512epochs-proud-star-1898.csv) |
| [rECCT](../conf/exp/recct-bch-63-45-ml-4m-2dB-aug.yaml) | 611K | 4M ML samples at 2 dB, with augmentation | 512 | [results](../log/test/recct-bch-63-45-ml-4m-2dB-aug-512epochs-fiery-plant-1871.csv) |

## eBCH (32, 16, 8)

Code file: [`ebch.32.16.mat`](../data/codes/ebch.32.16.mat). Dataset: [`train-ml-4M-3dB.mat`](https://sdrive.cnrs.fr/s/fx7kN9s5MwZfi35) (see [Pre-computed dataset format and download](training.md#pre-computed-dataset-format-and-download)); the 16M variant is currently not available for download (data augmentation can do equally well).

### Experiments with a pre-computed dataset

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [rECCT](../conf/exp/recct-ebch-32-16-ml-4m-3dB-aug.yaml) | 153K | 4M ML samples at 3 dB, with augmentation | 256 | [results](../log/test/recct-ebch-32-16-ml-4m-3dB-aug-256epochs-fearless-leaf-1947.csv) |
| [rECCT](../conf/exp/recct-ebch-32-16-ml-16m-3dB.yaml) | 153K | 16M ML samples at 3 dB | 128 | [results](../log/test/recct-ebch-32-16-ml-16m-3dB-128epochs-genial-sponge-1922.csv) |

## Reed-Muller (32, 16, 8)

Code file: [`rm.32.16.mat`](../data/codes/rm.32.16.mat). Dataset: [`train-ml-4M-3dB.mat`](https://sdrive.cnrs.fr/s/waykmQteWx5RZPn) (see [Pre-computed dataset format and download](training.md#pre-computed-dataset-format-and-download)). The RM code is non-systematic and ships with a reverse-encoding matrix `Ginv`; augmentation uses `GenericPerms`. Permutations file: [`perms.rm.32.mat`](../data/perms/perms.rm.32.mat). 

### Experiments with a pre-computed dataset

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [rECCT](../conf/exp/recct-rm-32-16-ml-4m-3dB.yaml) | 153K | 4M ML samples at 3 dB | 128 | [results](../log/test/recct-rm-32-16-ml-4m-3dB-128epochs-exalted-plant-1940.csv) |
| [rECCT](../conf/exp/recct-rm-32-16-ml-4m-3dB-aug.yaml) | 153K | 4M ML samples at 3 dB, with augmentation | 128 | [results](../log/test/recct-rm-32-16-ml-4m-3dB-aug-128epochs-vocal-paper-2012.csv) |

## QC-LDPC (96, 48, 10) from RPTU

Code file: [`ldpc.rptu.96.48.mat`](../data/codes/ldpc.rptu.96.48.mat).

### Experiments with on-demand data

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [ECCT](../conf/exp/ecct-ldpc-rptu-96-48-on-demand-3dB.yaml) | 2M | on-demand at 3 dB | 512 | [results](../log/test/ecct-ldpc-rptu-96-48-on-demand-3dB-512epochs-fiery-tree-1917.csv) |
| [rECCT](../conf/exp/recct-ldpc-rptu-96-48-on-demand-3dB.yaml) | 632K | on-demand at 3 dB | 512 | [results](../log/test/recct-ldpc-rptu-96-48-on-demand-3dB-512epochs-zany-star-1923.csv) |

## Polar (128, 64, 8) from RPTU

Code file: [`polar.rptu.128.64.mat`](../data/codes/polar.rptu.128.64.mat). Dataset for the dataset-based experiment below: [`train-ml-4M-3dB.mat`](https://sdrive.cnrs.fr/s/sdkG86Terq5KCLr) (see [Pre-computed dataset format and download](training.md#pre-computed-dataset-format-and-download)). The Polar code is non-systematic and ships with a reverse-encoding matrix `Ginv`; augmentation uses `GenericPerms`. Permutations file: [`perms.polar.128.mat`](../data/perms/perms.polar.128.mat). 

The three rECCT on-demand / dataset experiments below correspond to the three-step training recipe described in the [README highlight](../README.md#-why-sbnd).

### Experiments with on-demand data

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [ECCT](../conf/exp/ecct-polar-rptu-128-64-on-demand-4dB.yaml) | 2M | on-demand at 4 dB | 256 | [results](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967.csv) (+ [sb4](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967-sb4.csv), [sb8](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967-sb8.csv), [tta4](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967-tta4.csv), [tta8](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967-tta8.csv), [tta16](../log/test/ecct-polar-rptu-128-64-on-demand-4dB-256epochs-devout-eon-1967-tta16.csv)) |
| [rECCT](../conf/exp/recct-polar-rptu-128-64-on-demand-4dB.yaml) | 652K | on-demand at 4 dB (step 1) | 512 | [results](../log/test/recct-polar-rptu-128-64-on-demand-4dB-512epochs-atomic-yogurt-1978.csv) |
| [rECCT](../conf/exp/recct-polar-rptu-128-64-on-demand-5dB.yaml) | 652K | on-demand at 5 dB (step 3) | 64 | [results](../log/test/recct-polar-rptu-128-64-on-demand-5dB-64epochs-carbonite-cruiser-1989.csv) (+ [sb8](../log/test/recct-polar-rptu-128-64-on-demand-5dB-64epochs-carbonite-cruiser-1989-sb8.csv), [tta8](../log/test/recct-polar-rptu-128-64-on-demand-5dB-64epochs-carbonite-cruiser-1989-tta8.csv)) |

### Experiments with a pre-computed dataset

| Decoder | Size | Training data | Epochs | Eval log |
| --- | --- | --- | --- | --- |
| [rECCT](../conf/exp/recct-polar-rptu-128-64-ml-4m-3dB-aug.yaml) | 652K | 4M ML samples at 3 dB, with augmentation (step 2) | 256 | [results](../log/test/recct-polar-rptu-128-64-ml-4m-3dB-aug-256epochs-sith-nerf-herder-1986.csv) |
