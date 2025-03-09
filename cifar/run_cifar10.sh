# export PYTHONPATH= 
source ~/.bashrc
conda activate cotta-variations
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/source.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/norm.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/tent.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta_selftrain.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta_poly.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta_kl.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta_cosine.yaml

