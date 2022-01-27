import warnings
warnings.filterwarnings('ignore')

import torch
import torch.distributions as dist
import numpy as np
from models import models, resnet, wideresnet
from bayes.laplace import dla
from bayes.vb import llvb
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
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)

path = f'./pretrained_models'
path_ood = path + ('/'+args.ood_data if args.ood_data != 'imagenet' else '')

train_loader = dl.CIFAR10(train=True, augm_flag=False)
val_loader, test_loader = dl.CIFAR10(train=False, val_size=2000)
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

ood_loader = dl.ImageNet32(train=True, dataset='CIFAR10')

num_classes = 10
data_shape = [3, 32, 32]

method_types = ['MAP', 'DE', 'DLA', 'LLVB']
distortion_types = dl.CorruptedCIFAR10Dataset.distortions
severity_levels = range(1, 6)  # 1 ... 5

dla_statedict = None
dla_ood_statedict = None

tab_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_mmc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_ece = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_brier = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tab_loglik = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def load_model(model_name, type='plain'):
    assert type in ['plain', 'dirlik', 'noneclass', 'mixed', 'oe']

    if type in ['dirlik', 'noneclass', 'mixed', 'oe']:
        _path = path_ood
    else:
        _path = path

    depth = 16
    widen_factor = 4

    if model_name == 'DE':
        model = []
        for k in range(1, 6):  # 1...5
            _model = wideresnet.WideResNet(depth, widen_factor, num_classes)
            _model.cuda()
            _model.load_state_dict(torch.load(f'{_path}/DE/CIFAR10_{type}_{k}.pt'))
            _model.eval()

            model.append(_model)
    else:
        n_classes = num_classes+1 if type == 'noneclass' else num_classes
        model = wideresnet.WideResNet(depth, widen_factor, n_classes)

        if model_name == 'LLVB':
            model = llvb.LLVB(model)

        model.cuda()
        model.load_state_dict(torch.load(f'{_path}/CIFAR10_{type}{"_llvb" if model_name == "LLVB" else ""}.pt'))
        model.eval()

    return model


def predict_(test_loader, model, model_name, params=None):
    if model_name == 'DLA':
        py = evalutil.predict_laplace(test_loader, model, n_samples=20)
    elif model_name == 'LLVB':
        py = evalutil.predict_llvb(test_loader, model, n_samples=200)
    elif model_name == 'DE':
        py = evalutil.predict_ensemble(test_loader, model)
    else:  # MAP
        py = evalutil.predict(test_loader, model)

    return py.cpu().numpy()


def evaluate(model_name, type='plain', verbose=True):
    assert model_name in method_types

    model_ = load_model(model_name, type)
    params = None
    model_str = f'{model_name}-{type}'

    if verbose:
        print(f'Evaluating {model_str}')

    if model_name == 'DLA':
        USE_OOD = type in ['noneclass', 'dirlik', 'mixed', 'oe']

        if USE_OOD:
            _path = path_ood
        else:  # plain
            _path = path

        model = dla.DiagLaplace(model_)
        model.load_state_dict(torch.load(f'{_path}/CIFAR10_{type}_dla.pt'))
    else:
        model = model_

    # For all distortions, for all severity
    for d in distortion_types:
        for s in severity_levels:
            shift_loader = dl.CorruptedCIFAR10(d, s)
            py_shift = predict_(shift_loader, model, model_name, params=params)

            if type == 'noneclass':
                py_shift = py_shift[:, :-1]  # Ignore the none class

            targets = torch.cat([y for x, y in shift_loader], dim=0).numpy()

            tab_acc[model_str][d][s].append(evalutil.get_acc(py_shift, targets))
            tab_mmc[model_str][d][s].append(evalutil.get_mmc(py_shift))
            tab_ece[model_str][d][s].append(evalutil.get_calib(py_shift, targets)[0])
            tab_brier[model_str][d][s].append(evalutil.get_brier(py_shift, targets))
            tab_loglik[model_str][d][s].append(evalutil.get_loglik(py_shift, targets))


evaluate('MAP', 'plain')
evaluate('MAP', 'oe')
evaluate('DE', 'plain')

evaluate('DLA', 'plain')
evaluate('DLA', 'noneclass')
evaluate('DLA', 'dirlik')
evaluate('DLA', 'mixed')
evaluate('DLA', 'oe')

evaluate('LLVB', 'plain')
evaluate('LLVB', 'noneclass')
evaluate('LLVB', 'dirlik')
evaluate('LLVB', 'mixed')
evaluate('LLVB', 'oe')

# Save results
dir_name = f'results/CIFAR-10-C/'
dir_name += f'{args.ood_data}'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# To convert defaultdict to dict
import json

np.save(f'{dir_name}/mmc', json.loads(json.dumps(tab_mmc)))
np.save(f'{dir_name}/acc', json.loads(json.dumps(tab_acc)))
np.save(f'{dir_name}/ece', json.loads(json.dumps(tab_ece)))
np.save(f'{dir_name}/brier', json.loads(json.dumps(tab_brier)))
np.save(f'{dir_name}/loglik', json.loads(json.dumps(tab_loglik)))
