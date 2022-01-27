import torch
from torch import optim
from models.models import LeNetMadry
from models import resnet, wideresnet
from util.evaluation import *
from util.misc import dirichlet_nll_ood, label_smoothing
import util.dataloaders as dl
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os

from bayes.vb import llvb

from torch.cuda import amp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='CIFAR10')
parser.add_argument('--method', choices=['dirlik', 'noneclass', 'mixed', 'oe', 'plain'], default='plain')
parser.add_argument('--tau', type=float, default=0.1, help='Prior tempering parameter. Use 0.001 for CIFAR100 with `dirlik`.')
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
parser.add_argument('--n_samples', type=int, default=5, help='# of MC samples for ELBO')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

batch_size = 128
n_epochs = 100
EPS = 1e-30
path = './pretrained_models'

OOD_TRAINING = args.method in ['dirlik', 'noneclass', 'mixed', 'oe']

if OOD_TRAINING:
    path += f'/{args.ood_data}' if args.ood_data != 'imagenet' else ''

model_suffix = f'_{args.method}'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False)
ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

if args.ood_data == 'imagenet':
    ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

print(len(train_loader.dataset), len(test_loader.dataset))

num_classes = 100 if args.dataset == 'CIFAR100' else 10

if args.method == 'noneclass':
    num_classes += 1

prior_prec = 5e-4 * len(train_loader.dataset)

if args.method != 'plain':
    prior_prec *= 2  # Effective dataset size is doubled due to OOD data

prior_var = 1/prior_prec

if args.dataset in ['MNIST', 'FMNIST']:
    model = llvb.LLVB(LeNetMadry(num_classes), prior_var)

    # No weight decay for variational parameters
    param_groups = [
        {'params': model.base_model.parameters(), 'lr': 1e-3, 'weight_decay': 5e-4},
        {'params': model.var_params, 'lr': 1e-3, 'weight_decay': 0}
    ]

    opt = optim.Adam(param_groups)
else:
    base_model = wideresnet.WideResNet(16, 4, num_classes)
    model = llvb.LLVB(base_model, prior_var)

    # No weight decay for variational parameters
    param_groups = [
        {'params': model.base_model.parameters(), 'lr': 0.1, 'weight_decay': 5e-4},
        {'params': model.var_params, 'lr': 0.1, 'weight_decay': 0}
    ]

    opt = optim.SGD(param_groups, momentum=0.9, nesterov=True)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
pbar = trange(n_epochs)

## For automatic-mixed-precision
scaler = amp.GradScaler()

for epoch in pbar:
    if args.ood_data == 'imagenet':
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))
    elif args.ood_data == 'smooth':
        ood_loader = dl.Noise(train=True, dataset=args.dataset, size=len(train_loader.dataset), batch_size=batch_size)

    running_elbo, running_dkl = 0, 0

    for batch_idx, ((x_in, y_in), (x_out, _)) in enumerate(zip(train_loader, ood_loader)):
        opt.zero_grad()

        m = len(x_in)  # Batch size
        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)

        if OOD_TRAINING:
            x_out = x_out[:m].cuda(non_blocking=True)  # To make the length the same as x_in
            x = torch.cat([x_in, x_out], dim=0)

            if args.method == 'noneclass':
                # Last class, zero-indexed
                y_out = (num_classes-1) * torch.ones(m, device='cuda').long()
            else:
                y_out = 1/num_classes * torch.ones(m, num_classes, device='cuda').float()

            if args.method == 'dirlik':
                y_in = F.one_hot(y_in, num_classes).float()
                y_in = label_smoothing(y_in, eps=EPS)
        else:
            x = x_in

        with amp.autocast():
            outputs, dkl = model(x, args.n_samples)  # n_samples x batch_size x n_classes

            # Expected log-likelihood
            E_loglik = 0

            for s in range(args.n_samples):
                if args.method == 'dirlik':
                    prec = 100
                    probs = torch.softmax(outputs, -1)
                    loss = -dists.Dirichlet(prec*probs[s, :m, :].squeeze() + EPS).log_prob(y_in).sum()
                    loss += -dists.Dirichlet(prec*probs[s, m:, :].squeeze() + EPS).log_prob(y_out).sum()

                    if args.dataset == 'CIFAR100':
                        loss /= num_classes
                else:
                    loss = criterion(outputs[s, :m, :].squeeze(), y_in)  # In-distribution loss

                    if args.method == 'noneclass':
                        loss += criterion(outputs[s, m:, :], y_out)
                    elif args.method == 'oe':
                        # Sum all log-probs directly (0.5 following Hendrycks)
                        loss += -0.5*1/num_classes*torch.log_softmax(outputs[s, m:, :], 1).sum()
                    elif args.method == 'mixed':
                        loss += dirichlet_nll_ood(outputs[s, m:, :], prec=1, num_classes=num_classes)

                E_loglik += 1/args.n_samples * loss

            # The scaling 1/len(x) is so that the gradient is averaged---nicer for optim.
            loss = 1/m * (E_loglik + args.tau*dkl)

        scaler.scale(loss).backward()

        if args.method == 'dirlik' and args.dataset != 'CIFAR100':
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        running_elbo = 0.9*running_elbo - 0.1*len(x)*loss.item()  # loss is neg. avg. ELBO
        running_dkl = 0.9*running_dkl + 0.1*dkl.item()

    pred = predict_llvb(test_loader, model, n_samples=20).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100

    pbar.set_description(f'[Ep.: {epoch+1}; ELBO: {running_elbo:.1f}; KL: {running_dkl:.1f}; acc: {acc_val:.1f}]')


if not os.path.exists(path):
    os.makedirs(path)

print(f'{path}/{args.dataset}{model_suffix}_llvb.pt')
torch.save(model.state_dict(), f'{path}/{args.dataset}{model_suffix}_llvb.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{path}/{args.dataset}{model_suffix}_llvb.pt'))
model.eval()

print()

## In-distribution
py_in = predict_llvb(test_loader, model, n_samples=100).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
