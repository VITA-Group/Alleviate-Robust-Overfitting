# Robust Overfitting may be mitigated by properly learned smoothening

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for this paper [Robust Overfitting may be mitigated by properly learned smoothing](https://openreview.net/forum?id=qZzy5urZw9)

Tianlong Chen\*, Zhenyu Zhang\*, Sijia Liu, Shiyu Chang, Zhangyang Wang

## Overview

To alleviate the intriguing problem of robust overfitting, we investigate two empirical means to inject more learned smoothening during adversarial training (**AT**): one leveraging knowledge distillation (**KD**) and self-training to smooth the logits, the other performing stochastic weight averaging (**SWA**) to smooth the weights 

Highlights:

- **Smoothening mitigates robust overfitting:**  After adopting KD and SWA in AT, we mitigated robust overfitting and achieve a better trade-off between standard test accuracy and robustness.
- **Rich ablation experiments**:  We conducted plenty of ablation experiments and visualizations to investigate the reason why robust overfitting may be mitigated by these smoothening approaches.

## Experiment Results

**Training with KD and SWA to mitigate robust overfitting**

![](https://raw.githubusercontent.com/VITA-Group/Alleviate-Robust-Overfitting/main/Figs/flatness.png)

**Flattening the rugged input space** 

![](https://raw.githubusercontent.com/VITA-Group/Alleviate-Robust-Overfitting/main/Figs/flatness.png)

## Prerequisites

- pytorch 1.5.1
- torchvision 0.6.1 
- advertorch 0.2.3

## Usage

**Standard Training:**

```
python -u main_std.py \
	--data [dataset direction] \ 
	--dataset cifar10 \
	--arch resnet18 \
	--save_dir std_cifar10_resnet18 
```

**PGD Adversarial Training:**

```
python -u main_adv.py \
	--data [dataset direction] \ 
	--dataset cifar10 \
	--arch resnet18 \
	--save_dir AT_cifar10_resnet18 
```

**Adversarial Training with KD&SWA:**

```
python -u main_adv.py \
	--data [dataset direction] \ 
	--dataset cifar10 \
	--arch resnet18 \
	--save_dir KDSWA_cifar10_resnet18 \
	--swa \
	--lwf \
	--t_weight1 pretrained_models/cifar10_resnet18_std_SA_best.pt \
	--t_weight2 pretrained_models/cifar10_resnet18_adv_RA_best.pt
```

**Testing under PGD-20 Linf eps=8/255** :

```
python -u main_adv.py \
	--data [dataset direction] \
	--dataset cifar10 \
	--arch resnet18 \
	--eval \
	--pretrained pretrained_models/**.pt \
	--swa #if test with swa_model
```

## Citation

```

```

