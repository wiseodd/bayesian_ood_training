import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import os
import pickle
import argparse
import math
import time
from bisect import bisect_left
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchtext
from torchtext import data
from torchtext import datasets
from util import dataloaders as dl
import util.evaluation as evalutil
from models import rnn
from bayes.vb import llvb
from bayes.laplace import llla
import tqdm
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['sst', '20ng', 'trec'], default='sst')
parser.add_argument('--compute_hessian', default=False, action='store_true')
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(9999)
np.random.seed(9999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

dset_method = dl.datasets_dict[args.dataset.upper()]
test_iter, train_data, num_classes, vocab_size = dset_method(train=False, return_infos=True)
print(len(train_data), num_classes, vocab_size)
targets = torch.cat([(batch.label - 1) for batch in iter(test_iter)]).cpu().numpy()

ood_dset_names = ['SNLI', 'Multi30k', 'WMT16']
ood_iters = {}
for name in ood_dset_names:
    ood_iters[name] = dl.datasets_dict[name](vocab_data=train_data)

method_types = ['MAP', 'DE', 'LLLA', 'LLVB']
types = ['plain', 'oe', 'noneclass', 'dirlik', 'mixed']
path = './pretrained_models/NLP'

tab_mmc = defaultdict(lambda: defaultdict(list))
tab_auroc = defaultdict(lambda: defaultdict(list))
tab_auprc = defaultdict(lambda: defaultdict(list))
tab_fpr95 = defaultdict(lambda: defaultdict(list))
tab_acc = defaultdict(list)
tab_cal = defaultdict(list)


# Model loading

def load_model(model_name, type='plain'):
    assert model_name in method_types
    assert type in types

    if model_name != 'DE':
        model = rnn.GRUClassifier(num_classes, vocab_size).cuda()

        if model_name == 'LLVB':
            model = llvb.LLVB(model).cuda()
            model_name = 'vb' + ('ood' if type == 'ood' else '')
        else:
            model_name = type

        model.load_state_dict(torch.load(f'{path}/{args.dataset}_{model_name}.pt'))
        model.eval()
    else:  # DeepEnsemble
        model = []
        params = torch.load(f'{path}/{args.dataset}_de.pt')

        for param in params:
            m = rnn.GRUClassifier(num_classes, vocab_size).cuda()
            m.load_state_dict(param)
            m.eval()
            model.append(m)

    return model


@torch.no_grad()
def predict(test_iter, model, model_name, params=None):
    probs = []

    for batch_idx, batch in enumerate(iter(test_iter)):
        try:
            inputs = (batch.text.t()).cuda()
        except AttributeError:
            # For SNLI
            inputs = (batch.hypothesis.t()).cuda()

        if model_name != 'DE':
            if model_name == 'MAP':
                py = torch.softmax(model(inputs), 1)
            elif model_name == 'LLLA':
                py = llla.predict_batch(inputs, model, *params, n_samples=20)
            else:  # LLVB
                n_samples = 20
                py = model(inputs, n_samples)

                py_mc = 0
                for s in range(n_samples):
                    py_mc += 1/n_samples * torch.softmax(py[s], 1)
                py = py_mc
        else:
            py = 0
            for k in range(5):
                py += 1/5 * torch.softmax(model[k](inputs), 1)

        probs.append(py)

    return torch.cat(probs, 0).cpu().numpy()


def evaluate(model_name, type='plain', verbose=True, ood_with_noneclass=False):
    assert model_name in method_types

    model = load_model(model_name, type)
    params = None
    model_str = f'{model_name}-{type}'

    if type == 'noneclass' and ood_with_noneclass:
        model_str += '-extra'

    if model_name == 'LLLA':
        if args.compute_hessian:
            train_loader = dset_method(train=True)
            use_ood = type in ['noneclass', 'dirlik', 'mixed', 'oe']
            ood_loader = iter(dl.WikiText2(train=True, vocab_data=train_data)) if use_ood else None
            hessians = llla.get_hessian(model, iter(train_loader), ood_loader, num_classes, type)
            interval = torch.logspace(-3, 1, 100)
            var0 = llla.gridsearch_var0(model, hessians, train_loader, interval, num_classes)
            print(var0)
            params = llla.estimate_variance(var0, hessians)
            torch.save(params, f'{path}/{args.dataset}_llla_{type}.pt')

            print(f'Hessian for {model_str} saved!')
            return

        params = torch.load(f'{path}/{args.dataset}_llla_{type}.pt')

    USE_NONE_CLASS = type == 'noneclass'

    py_in = predict(test_iter, model, model_name, params=params)

    acc = np.mean(np.argmax(py_in, 1) == targets)*100
    ece, _ = evalutil.get_calib(py_in if not USE_NONE_CLASS else py_in[:, :-1], targets)

    if not ood_with_noneclass and type == 'noneclass':
        py_in = py_in[:, :-1]  # Exclude the none class
        USE_NONE_CLASS = False

    mmc = evalutil.get_mmc(py_in, USE_NONE_CLASS)

    tab_mmc[model_str][args.dataset.upper()].append(mmc)
    tab_acc[model_str].append(acc)
    tab_cal[model_str].append(ece)

    if verbose:
        print(f'[In, {model_str}] Acc: {acc:.1f}; ECE: {ece:.1f}; MMC: {mmc:.3f}')

    if not args.compute_hessian:
        for ood_name, ood_iter in ood_iters.items():
            py_out = predict(ood_iter, model, model_name, params)

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
pbar = range(args.repeat) if args.repeat == 1 else tqdm.trange(args.repeat)

for i in pbar:
    if not args.compute_hessian:
        evaluate('MAP', 'plain', verbose)
        evaluate('MAP', 'oe', verbose)
        evaluate('DE', 'plain', verbose)
        # evaluate('LLVB', 'plain', verbose)
        # evaluate('LLVB', 'ood', verbose)

    evaluate('LLLA', 'plain', verbose)
    evaluate('LLLA', 'noneclass', verbose)
    evaluate('LLLA', 'noneclass', verbose, ood_with_noneclass=True)
    evaluate('LLLA', 'dirlik', verbose)
    evaluate('LLLA', 'mixed', verbose)
    evaluate('LLLA', 'oe', verbose)

    if verbose and i < args.repeat-1:
        print('-------------------------------------------')
        print()


# Save results if repeated
if args.repeat > 1:
    dir_name = f'results/NLP/{args.dataset.upper()}'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.save(f'{dir_name}/mmc', dict(tab_mmc))
    np.save(f'{dir_name}/auroc', dict(tab_auroc))
    np.save(f'{dir_name}/auprc', dict(tab_auprc))
    np.save(f'{dir_name}/fpr95', dict(tab_fpr95))
    np.save(f'{dir_name}/acc', dict(tab_acc))
    np.save(f'{dir_name}/cal', dict(tab_cal))
