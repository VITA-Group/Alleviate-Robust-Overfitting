# Robust Overfitting may be mitigated by properly learned smoothening

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for this paper Robust Overfitting may be mitigated by properly learned smoothing

Tianlong Chen\*, Zhenyu Zhang\*, Sijia Liu, Shiyu Chang, Zhangyang Wang

Test environment

​	pytorch = 1.5.1

​	torchvision = 0.6.1

​	advertorch = 0.2.3

​	matplotlib = 3.3.1

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

