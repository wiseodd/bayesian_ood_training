import torch
from torch import optim
from torch import distributions as dists
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
import math
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='CIFAR10')
parser.add_argument('--method', choices=['dirlik', 'noneclass', 'mixed', 'oe', 'de', 'plain'], default='plain')
parser.add_argument('--large', action='store_true', default=False)
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'smooth'])
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

if args.method == 'de':
    # Obtain different randseed
    args.randseed = np.random.randint(0, 9999)
    print(f'Random seed: {args.randseed}')

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

batch_size = 128
n_epochs = 100
EPS = 1e-30
path = './pretrained_models' + f'{"/large" if args.large else ""}'

OOD_TRAINING = args.method in ['dirlik', 'noneclass', 'mixed', 'oe', 'de']

if OOD_TRAINING:
    path += f'/{args.ood_data}' if args.ood_data != 'imagenet' else ''

model_suffix = f'_{args.method}'

if args.method == 'de':
    model_suffix += f'_{args.randseed}'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10

if args.method == 'noneclass':
    num_classes += 1

if args.ood_data == 'imagenet':
    ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

print(len(train_loader.dataset), len(test_loader.dataset))

if args.dataset in ['MNIST', 'FMNIST']:
    model = LeNetMadry(num_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
else:
    depth = 40 if args.large else 16
    widen_factor = 2 if args.large else 4
    model = wideresnet.WideResNet(depth, widen_factor, num_classes)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))
pbar = trange(n_epochs)

## For automatic-mixed-precision
scaler = amp.GradScaler()

# Timing stuff
timing_start = torch.cuda.Event(enable_timing=True)
timing_end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
timing_start.record()

for epoch in pbar:
    # Get new data every epoch to avoid overfitting to noises
    if args.ood_data == 'imagenet':
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))
    elif args.ood_data == 'smooth':
        ood_loader = dl.Noise(train=True, dataset=args.dataset, size=len(train_loader.dataset), batch_size=batch_size)

    train_loss = 0

    for batch_idx, ((x_in, y_in), (x_out, _)) in enumerate(zip(train_loader, ood_loader)):
        model.train()
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
                # Label smoothing is needed since there is the term log(y) in the Dirichlet loglik
                y_in = F.one_hot(y_in, num_classes).float()
                y_in = label_smoothing(y_in, eps=EPS)
        else:
            x = x_in

        with amp.autocast():
            outputs = model(x).squeeze()

            if args.method == 'dirlik':
                prec = 100
                probs = torch.softmax(outputs, 1)
                loss = -dists.Dirichlet(prec*probs[:m] + EPS).log_prob(y_in).mean()
                loss += -dists.Dirichlet(prec*probs[m:] + EPS).log_prob(y_out).mean()

                if args.dataset == 'CIFAR100':
                    loss /= num_classes
            else:
                loss = criterion(outputs[:m], y_in)  # In-distribution loss

                if args.method == 'noneclass':
                    loss += criterion(outputs[m:], y_out)
                elif args.method == 'mixed':
                    loss += dirichlet_nll_ood(outputs[m:], prec=1, num_classes=num_classes)
                elif args.method == 'oe':
                    # Sum all log-probs directly (0.5 following Hendrycks)
                    loss += -0.5*1/num_classes*torch.log_softmax(outputs[m:], 1).mean()

        scaler.scale(loss).backward()

        if args.method in ['dirlik'] and args.dataset != 'CIFAR100':
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = predict(test_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )

# Timing stuff
timing_end.record()
torch.cuda.synchronize()
timing = timing_start.elapsed_time(timing_end)/1000
np.save(f'results/timing_{args.method}_{args.dataset.lower()}_711', timing)

if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/{args.dataset}{model_suffix}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{path}/{args.dataset}{model_suffix}.pt'))
model.eval()

print()

## In-distribution
py_in = predict(test_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
