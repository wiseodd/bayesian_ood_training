#!/bin/bash

## ------------------------------
## OOD DETECTION
## ------------------------------

python eval_OOD.py --dataset MNIST --compute_hessian
python eval_OOD.py --dataset FMNIST --compute_hessian
python eval_OOD.py --dataset SVHN --compute_hessian
python eval_OOD.py --dataset CIFAR10 --compute_hessian
python eval_OOD.py --dataset CIFAR100 --compute_hessian

python eval_OOD.py --dataset MNIST --repeat 5
python eval_OOD.py --dataset FMNIST --repeat 5
python eval_OOD.py --dataset CIFAR10 --repeat 5
python eval_OOD.py --dataset CIFAR100 --repeat 5
python eval_OOD.py --dataset SVHN --repeat 5


## ------------------------------
## SMOOTH NOISE OOD
## ------------------------------

python eval_OOD.py --dataset MNIST --ood_data smooth --compute_hessian
python eval_OOD.py --dataset FMNIST --ood_data smooth --compute_hessian
python eval_OOD.py --dataset CIFAR10 --ood_data smooth --compute_hessian
python eval_OOD.py --dataset CIFAR100 --ood_data smooth --compute_hessian
python eval_OOD.py --dataset SVHN --ood_data smooth --compute_hessian

python eval_OOD.py --dataset MNIST --repeat 5 --ood_data smooth
python eval_OOD.py --dataset FMNIST --repeat 5 --ood_data smooth
python eval_OOD.py --dataset CIFAR10 --repeat 5 --ood_data smooth
python eval_OOD.py --dataset CIFAR100 --repeat 5 --ood_data smooth
python eval_OOD.py --dataset SVHN --repeat 5 --ood_data smooth


## ------------------------------
## DATASET SHIFT
## ------------------------------

python eval_MNIST-C.py --transform rotation
python eval_MNIST-C.py --transform translation
python eval_CIFAR10-C.py


## ---------------------------------------
## AUXILIARY RESULTS (DE, FLIPOUT, CSGHMC)
## ---------------------------------------

python eval_OOD.py --dataset CIFAR10 --repeat 5 --aux_models
python eval_OOD.py --dataset CIFAR100 --repeat 5 --aux_models
