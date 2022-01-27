import warnings
warnings.filterwarnings('ignore')
import sys
import torch
import torch.distributions as dist
import numpy as np
from models import models, resnet, wideresnet
from bayes.laplace import dla
from bayes.vb import llvb, bbb
from util import evaluation as evalutil
import util.dataloaders as dl
from math import *
from tqdm import tqdm, trange
import argparse
import pickle
import os, sys
from tqdm import tqdm, trange
import torch.utils.data as data_utils
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--compute_hessian', action='store_true', default=False)
parser.add_argument('--aux_models', action='store_true', default=False)
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100']

if args.aux_models and args.dataset in ['MNIST', 'FMNIST', 'SVHN']:
    print('Auxiliary models only available for CIFARs'); sys.exit(1)

path = f'./pretrained_models'
path_ood = path + ('/'+args.ood_data if args.ood_data != 'imagenet' else '')

train_loader = dl.datasets_dict[args.dataset](train=True, augm_flag=False)
val_loader, test_loader = dl.datasets_dict[args.dataset](train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

ood_loader = dl.ImageNet32(train=True, dataset=args.dataset)

num_classes = 100 if args.dataset == 'CIFAR100' else 10
data_shape = [1, 28, 28] if args.dataset == 'MNIST' else [3, 32, 32]

method_types = ['MAP', 'DE', 'DLA', 'LLVB', 'CSGHMC', 'BBB']
ood_noise_names = ['UniformNoise', 'Noise', 'FarAway']
ood_test_names = {
    'MNIST': ['FMNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'FMNIST': ['MNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D'],
    'CIFAR100': ['SVHN', 'LSUN', 'CIFAR10', 'FMNIST3D'],
}

ood_names = ood_test_names[args.dataset] + ood_noise_names
ood_test_loaders = {}

for ood_name in ood_test_names[args.dataset]:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](train=False)

for ood_name in ood_noise_names:
    ood_test_loaders[ood_name] = dl.datasets_dict[ood_name](dataset=args.dataset, train=False, size=10000)

tab_mmc = defaultdict(lambda: defaultdict(list))
tab_auroc = defaultdict(lambda: defaultdict(list))
tab_auprc = defaultdict(lambda: defaultdict(list))
tab_fpr95 = defaultdict(lambda: defaultdict(list))
tab_acc = defaultdict(list)
tab_cal = defaultdict(list)


def load_model(model_name, type='plain'):
    assert type in ['plain', 'dirlik', 'noneclass', 'mixed', 'oe']

    if type in ['dirlik', 'noneclass', 'mixed', 'oe']:
        _path = path_ood
    else:
        _path = path

    depth = 16
    widen_factor = 4
    n_classes = num_classes+1 if type == 'noneclass' else num_classes

    if model_name == 'DE':
        model = []

        if type == 'plain':
            state_dicts = [torch.load(f'{_path}/DE/{args.dataset}_plain_{k}.pt') for k in range(1, 6)]
        else:
            state_dicts = torch.load(f'{_path}/{args.dataset}_{type}_de.pt')

        for k in range(5):
            if args.dataset in ['MNIST', 'FMNIST']:
                _model = models.LeNetMadry()
            else:
                _model = wideresnet.WideResNet(depth, widen_factor, n_classes)

            _model.cuda()
            _model.load_state_dict(state_dicts[k])
            _model.eval()

            model.append(_model)
    elif model_name == 'CSGHMC':
        samples_state_dicts = torch.load(f'{_path}/{args.dataset}_{type}_csghmc.pt')
        model = []
        for state_dict in samples_state_dicts:
            _model = wideresnet.WideResNet(16, 4, n_classes, dropRate=0).cuda()
            _model.load_state_dict(state_dict)
            model.append(_model.eval())
    elif model_name == 'BBB':
        model = bbb.WideResNetBBB(16, 4, n_classes, estimator='flipout').cuda()
        model.load_state_dict(torch.load(f'{_path}/{args.dataset}_{type}_bbb_flipout.pt'))
        model.eval()
    else:
        if args.dataset in ['MNIST', 'FMNIST']:
            model = models.LeNetMadry(n_classes)
        else:
            model = wideresnet.WideResNet(depth, widen_factor, n_classes)

        if model_name == 'LLVB':
            model = llvb.LLVB(model)

        model.cuda()
        state_dict = torch.load(
            f'{_path}/{args.dataset}_{type}{"_llvb" if model_name == "LLVB" else ""}.pt'
        )
        model.load_state_dict(state_dict)
        model.eval()

    return model


def predict_(test_loader, model, model_name, params=None):
    if model_name == 'DLA':
        py = evalutil.predict_laplace(test_loader, model, n_samples=20)
    elif model_name == 'LLVB':
        py = evalutil.predict_llvb(test_loader, model, n_samples=200)
    elif model_name in ['DE', 'CSGHMC']:
        py = evalutil.predict_ensemble(test_loader, model)
    elif model_name == 'BBB':
        py = evalutil.predict(test_loader, model, n_samples=10)
    else:  # MAP
        py = evalutil.predict(test_loader, model)

    return py.cpu().numpy()


def evaluate(model_name, type='plain', verbose=True, ood_with_noneclass=False):
    assert model_name in method_types

    model_ = load_model(model_name, type)
    params = None
    model_str = f'{model_name}-{type}'

    if type == 'noneclass' and ood_with_noneclass:
        model_str += '-extra'

    if model_name == 'DLA':
        USE_OOD = type in ['noneclass', 'dirlik', 'mixed', 'oe']

        if USE_OOD:
            _path = path_ood
        else:  # plain
            _path = path

        model = dla.DiagLaplace(model_)

        if args.compute_hessian:
            model.get_hessian(train_loader, ood_loader=ood_loader if USE_OOD else None, mode=type)
            interval = torch.logspace(-6, -3, 100)
            var0 = model.gridsearch_var0(val_loader, interval, lam=1)
            print(var0)
            model.estimate_variance(var0)
            torch.save(model.state_dict(), f'{_path}/{args.dataset}_{type}_dla.pt')

            print(f'Hessian for {model_str} saved at {_path}/{args.dataset}_{type}_dla.pt!')
            return

        model.load_state_dict(torch.load(f'{_path}/{args.dataset}_{type}_dla.pt'))
    else:
        model = model_

    USE_NONE_CLASS = type == 'noneclass'

    py_in = predict_(test_loader, model, model_name, params)

    acc = np.mean(np.argmax(py_in, 1) == targets)*100
    ece, _ = evalutil.get_calib(py_in if not USE_NONE_CLASS else py_in[:, :-1], targets)

    if not ood_with_noneclass and type == 'noneclass':
        py_in = py_in[:, :-1]  # Exclude the none class
        USE_NONE_CLASS = False

    mmc = evalutil.get_mmc(py_in, USE_NONE_CLASS)

    tab_mmc[model_str][args.dataset].append(mmc)
    tab_acc[model_str].append(acc)
    tab_cal[model_str].append(ece)

    if verbose:
        print(f'[In, {model_str}] Acc: {acc:.1f}; ECE: {ece:.1f}; MMC: {mmc:.3f}')

    for ood_name, ood_test_loader in ood_test_loaders.items():
        py_out = predict_(ood_test_loader, model, model_name, params)

        if not ood_with_noneclass and type == 'noneclass':
            py_out = py_out[:, :-1]  # Exclude the none class

        mmc = evalutil.get_mmc(py_out, USE_NONE_CLASS)
        auroc = evalutil.get_auroc(py_in, py_out, USE_NONE_CLASS)
        auprc = evalutil.get_aupr(py_in, py_out, USE_NONE_CLASS)
        fpr95, _ = evalutil.get_fpr95(py_in, py_out, USE_NONE_CLASS)

        tab_mmc[model_str][ood_name].append(mmc)
        tab_auroc[model_str][ood_name].append(auroc)
        tab_auprc[model_str][ood_name].append(auprc)
        tab_fpr95[model_str][ood_name].append(fpr95)

        if verbose:
            print(f'[Out-{ood_name}, {model_str}] MMC: {mmc:.1f}; AUROC: {auroc:.1f}; '
                  + f'AUPRC: {auprc:.1f} FPR@95: {fpr95:.1f}')

    if verbose:
        print()


verbose = True if args.repeat == 1 else False
pbar = range(args.repeat) if args.repeat == 1 else trange(args.repeat)

for i in pbar:
    if not args.aux_models:
        if not args.compute_hessian:
            evaluate('MAP', 'plain', verbose)
            evaluate('MAP', 'oe', verbose)
            evaluate('DE', 'plain', verbose)

            evaluate('LLVB', 'plain', verbose)
            evaluate('LLVB', 'noneclass', verbose)
            evaluate('LLVB', 'noneclass', verbose, ood_with_noneclass=True)
            evaluate('LLVB', 'dirlik', verbose)
            evaluate('LLVB', 'mixed', verbose)
            evaluate('LLVB', 'oe', verbose)

        evaluate('DLA', 'plain', verbose)
        evaluate('DLA', 'noneclass', verbose)

        if not args.compute_hessian:
            evaluate('DLA', 'noneclass', verbose, ood_with_noneclass=True)

        evaluate('DLA', 'dirlik', verbose)
        evaluate('DLA', 'mixed', verbose)
        evaluate('DLA', 'oe', verbose)
    else:
        # Additional, auxiliary experiments for appendix
        evaluate('BBB', 'plain', verbose)
        evaluate('CSGHMC', 'plain', verbose)

        evaluate('DE', 'noneclass', verbose)
        evaluate('BBB', 'noneclass', verbose)
        evaluate('CSGHMC', 'noneclass', verbose)

    if verbose and i < args.repeat-1:
        print('-------------------------------------------')
        print()


# Save results if repeated
if args.repeat > 1:
    dir_name = f'results/OOD/'
    dir_name += f'{args.ood_data}/' + args.dataset

    if args.aux_models:
        dir_name += '/aux'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.save(f'{dir_name}/mmc', dict(tab_mmc))
    np.save(f'{dir_name}/auroc', dict(tab_auroc))
    np.save(f'{dir_name}/auprc', dict(tab_auprc))
    np.save(f'{dir_name}/fpr95', dict(tab_fpr95))
    np.save(f'{dir_name}/acc', dict(tab_acc))
    np.save(f'{dir_name}/cal', dict(tab_cal))
