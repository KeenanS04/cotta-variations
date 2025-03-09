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

## Classification Experiment
### CIFAR10-to-CIFAR10C-standard task
```bash
cd cifar
# This includes the comparison of all three methods as well as baseline
bash run_cifar10.sh 
```

Running the above gets the "raw" results for our experiment, in terms of predicted labels and true labels for our experiment with 10,000 images.


# Citations
Authors: Nick Swetlin, Keenan Serrao, Ifunanya Okoroma, Ansh Mujral.

Heavily inspired by Qin et. al.'s [initial work on COTTA.](https://github.com/qinenergy/cotta)
Cite them:

```bibtex
@inproceedings{wang2022continual,
  title={Continual Test-Time Domain Adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
