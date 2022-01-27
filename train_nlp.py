# -*- coding: utf-8 -*-
"""
Trains a MNIST classifier.
"""

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
import torch.distributions as dists
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchtext
import torchtext.data as ttdata
from torchtext import datasets
from bayes.vb import llvb
from util.misc import dirichlet_nll_ood, label_smoothing
from models import rnn
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['sst', '20ng', 'trec'], default='sst')
parser.add_argument('--model', type=str, choices=['plain', 'oe', 'noneclass', 'dirlik', 'mixed', 'de', 'vb', 'vb_noneclass', 'vb_dirlik', 'vb_mixed', 'vb_oe'], default='plain')
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--n_samples', type=int, default=5, help='Num. MC samples for VB')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()


torch.manual_seed(args.randseed)
np.random.seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if args.dataset == 'sst':
    # set up fields
    TEXT = ttdata.Field(pad_first=True)
    LABEL = ttdata.Field(sequential=False)

    # make splits for data
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=False, train_subtrees=False,
        filter_pred=lambda ex: ex.label != 'neutral')

    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)

    # create our own iterator, avoiding the calls to build_vocab in SST.iters
    train_iter, val_iter, test_iter = ttdata.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, repeat=False)
elif args.dataset == '20ng':
    TEXT = ttdata.Field(pad_first=True, lower=True, fix_length=100)
    LABEL = ttdata.Field(sequential=False)

    train = ttdata.TabularDataset(path='./.data/20newsgroups/20ng-train.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    test = ttdata.TabularDataset(path='./.data/20newsgroups/20ng-test.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)

    train_iter = ttdata.BucketIterator(train, batch_size=args.batch_size, repeat=False)
    test_iter = ttdata.BucketIterator(test, batch_size=args.batch_size, repeat=False)
elif args.dataset == 'trec':
    # set up fields
    TEXT = ttdata.Field(pad_first=True, lower=True)
    LABEL = ttdata.Field(sequential=False)

    # make splits for data
    train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)

    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)

    # make iterators
    train_iter, test_iter = ttdata.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, repeat=False)


# OOD dataset---WikiText2
TEXT_custom = ttdata.Field(pad_first=True, lower=True)
custom_data = ttdata.TabularDataset(path='./.data/wikitext_reformatted/wikitext2_sentences',
                                  format='csv',
                                  fields=[('text', TEXT_custom)])

TEXT_custom.build_vocab(train.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_custom.vocab))
train_iter_out = ttdata.BucketIterator(custom_data, batch_size=args.batch_size, repeat=False)


num_classes = len(LABEL.vocab)-1
EPS = 1e-30

if args.model == 'noneclass':
    num_classes += 1

if args.model != 'de':
    model = rnn.GRUClassifier(num_classes, len(TEXT.vocab)).cuda()

    if 'vb' in args.model:
        prior_prec = 5e-4 * len(train)

        if 'ood' in args.model:
            prior_prec *= 2  # Effective dataset size is doubled due to OOD data

        prior_var = 1/prior_prec
        model = llvb.LLVB(model, prior_var).cuda()

    wd = 5e-4 if args.model == 'ood' else 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:  # DeepEnsemble
    models = [rnn.GRUClassifier(num_classes, len(TEXT.vocab)).cuda() for _ in range(5)]
    optimizers = [torch.optim.Adam(models[k].parameters(), lr=0.01) for k in range(5)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=args.epochs) for k in range(5)]


def train():
    if args.model != 'de':
        model.train()
    else:
        for k in range(5):
            models[k].train()

    for batch_idx, (batch, batch_out) in enumerate(zip(iter(train_iter), iter(train_iter_out))):
        inputs = batch.text.t().cuda()
        labels = (batch.label - 1).cuda()

        m = len(inputs)
        inputs_out = (batch_out.text.t())[:m].cuda()

        if args.model == 'noneclass':
            # Last class, zero-indexed
            y_out = (num_classes-1) * torch.ones(m, device='cuda').long()
        else:
            y_out = 1/num_classes * torch.ones(m, num_classes, device='cuda').float()

        if args.model == 'dirlik':
            # Label smoothing is needed since there is the term log(y) in the Dirichlet loglik
            y_in = F.one_hot(labels, num_classes).float()
            y_in = label_smoothing(y_in, eps=EPS)

        if args.model != 'de':
            if args.model in ['plain', 'oe', 'mixed', 'noneclass', 'dirlik']:
                if args.model == 'dirlik':
                    prec = 100
                    prob_in = torch.softmax(model(inputs), 1)
                    prob_out = torch.softmax(model(inputs_out), 1)
                    loss = -dists.Dirichlet(prec*prob_in + EPS).log_prob(y_in).mean()
                    loss += -dists.Dirichlet(prec*prob_out + EPS).log_prob(y_out).mean()
                else:
                    # In-distribution loss
                    loss = F.cross_entropy(model(inputs), labels)

                    logit_out = model(inputs_out)

                    if args.model == 'noneclass':
                        loss += F.cross_entropy(logit_out, y_out)
                    elif args.model == 'mixed':
                        # Averaged
                        loss += dirichlet_nll_ood(logit_out, prec=1, num_classes=num_classes).mean()
                    elif args.model == 'oe':
                        loss += -0.5*1/num_classes*torch.log_softmax(logit_out, 1).mean()
            elif 'vb_' in args.model:
                # Use 5 MC samples
                outputs, dkl = model(inputs, args.n_samples)  # n_samples x batch_size x n_classes

                # Expected log-likelihood
                E_loglik = 0

                for s in range(args.n_samples):
                    # In-distribution loss
                    loss = F.cross_entropy(outputs[s, :, :].squeeze(), labels, reduction='sum')

                    if 'ood' in args.model:
                        inputs_out = (batch_out.text.t()).cuda()
                        outputs_out, _ = model(inputs_out, args.n_samples)
                        # (Tempered) OOD loss (Dirichlet NLL)
                        loss += 1/num_classes * dirichlet_nll(outputs_out[s, :, :].squeeze(), num_classes=num_classes)

                    E_loglik += 1/args.n_samples * loss

                # The scaling 1/len(x) is so that the gradient is averaged---nicer for optim.
                loss = 1/args.batch_size * (E_loglik + 0.1*dkl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # DeepEnsemble
            for k in range(5):
                logits = models[k](inputs)
                loss = F.cross_entropy(logits, labels)
                optimizers[k].zero_grad()
                loss.backward()
                optimizers[k].step()

    if args.model != 'de':
        scheduler.step()
    else:
        for k in range(5):
            schedulers[k].step()

    return loss


def evaluate():
    if args.model != 'de':
        model.eval()
    else:
        for k in range(5):
            models[k].eval()

    num_examples = 0
    correct = 0

    for batch_idx, batch in enumerate(iter(test_iter)):
        inputs = (batch.text.t()).cuda()
        labels = (batch.label - 1).cuda()

        if args.model != 'de':

            if 'vb' not in args.model:
                out = model(inputs)
            else:
                n_samples = 100
                out = model(inputs, n_samples)

                out_mc = 0
                for s in range(n_samples):
                    out_mc += 1/n_samples * torch.softmax(out[s], 1)
                out = out_mc
        else:
            out = 0
            for k in range(5):
                out += 1/5 * models[k](inputs)

        pred = out.max(1)[1]
        correct += pred.eq(labels).sum().data.cpu().numpy()

        num_examples += inputs.shape[0]

    return correct / num_examples


pbar = tqdm.trange(args.epochs)
for epoch in pbar:
    loss = train()
    acc = evaluate()
    pbar.set_description(f'[Epoch-{epoch+1}; loss: {loss:.3f}; test acc: {acc*100:.1f}]')

# Save
path = './pretrained_models/NLP'
if not os.path.exists(path):
    os.makedirs(path)

if args.model == 'de':
    torch.save([models[k].state_dict() for k in range(5)], f'{path}/{args.dataset}_{args.model}.pt')
else:
    torch.save(model.state_dict(), f'{path}/{args.dataset}_{args.model}.pt')

print('Model saved')
