# cotta-variations
Testing variations of Continual Test-Time Adaptation (CoTTA) for robustness.

## Prerequisites

Please create and activate the following conda environment to recreate our experiment. Note that a Linux distribution is necessary to install the required Anaconda packages. Note that an NVIDIA GPU with CUDA capabilities is also necessary. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cotta 
```

## Classification Experiment
### CIFAR10-to-CIFAR10C Task

Once the cotta-variations environment is created, the following command can be used to recreate our experiment.
```bash
cd cifar
# This includes the comparison of all three methods as well as baseline
bash run_cifar10.sh 
```

Running the above outputs 8 files for a single model to be adapted -- one file of raw results for each loss function (which needs to be averaged over the file to recreate one cell from our tables). By default, these 8 files correspond to one column in our results table, not the entire table itself. Since writing a single command to run the entire experiment 8 times (one iteration for each model) would take far too long in a single sitting, we will instead instruct how to use configurations to recreate our paper results incrementally if desired.

### Editing Configurations
Before running ```bash run_cifar10.sh```, one should know how to edit configurations for our entire experiment. The file ```cifar10/conf.py``` contains all the configurations one would need to fully adjust this experiment.

There are two configurations in ```cifar10/conf.py``` that are especially important:

- ```_C.CORRUPTION.NUM_EX``` is the size of the CIFAR10C test dataset evaluate. By default, this value is 10000 (the size of the full dataset). **We recommend lowering this value to 100 to significantly expedite the experiment if one is simply exploring our repo.**
- ```_C.MODEL.ARCH``` is the model to evaluate. By default, this value is 'Standard'. To recreate our entire results, one would need to run ```bash run_cifar10.sh``` 7 times, once for each of the following values for ```_C.MODEL.ARCH```:

These models (M1 - M7) are sorted by size for convenience.
```
['Kireev2021Effectiveness_Gauss50percent',
'Kireev2021Effectiveness_RLAT',
'Kireev2021Effectiveness_RLATAugMix',
'Standard',
'Hendrycks2020AugMix_ResNeXt',
'Addepalli2022Efficient_WRN_34_10',
'Hendrycks2020AugMix_WRN']
```


# Citations
Authors: Nick Swetlin, Keenan Serrao, Ifunanya Okoroma, Ansh Mujral.

Mentor: Jun Kun Wang

Cite us:
```bibtex
@inproceedings{nkia2025cottavars,
  title={Investigating CoTTA: Validating Real-Time Neural Network
Adaptations},
  author={Swetlin, Nick and Serrao, Keenan and Okoroma, Ifunanya and Mujral, Ansh},
  year={2025}
}
```

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
