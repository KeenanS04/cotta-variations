# cotta-variations
Testing variations of Continual Test-Time Adaptation (CoTTA) for robustness.

## Prerequisites

Please create and activate the following conda environment to recreate our experiment. Note that a Linux distribution is necessary to install the required Anaconda packages. Note that an NVIDIA GPU with CUDA capabilities is necessary. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cotta 
```

## Classification Experiments
### CIFAR10-to-CIFAR10C-standard task
```bash
# Tested on RTX2080TI
cd cifar
# This includes the comparison of all three methods as well as baseline
bash run_cifar10.sh 
```

### CIFAR100-to-CIFAR100C task
```bash
# Tested on RTX3090
bash run_cifar100.sh
```

# Citations
Authors: Nick Swetlin, Keenan Serrao, Ifunanya Okoroma, Ansh Mujral.

Heavily inspired by Qin et. al.'s [initial work on COTTA.](https://github.com/qinenergy/cotta)
Cite them:
@inproceedings{wang2022continual,
  title={Continual Test-Time Domain Adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
