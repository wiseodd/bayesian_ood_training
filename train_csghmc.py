import os
import torch
import torch.nn.functional as F
from torch import optim
from models.wideresnet import WideResNet
from bayes.mcmc.csghmc import CSGHMCTrainer
import util.evaluation as evalutil
import util.dataloaders as dl
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import math
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=['SVHN', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--method', choices=['noneclass', 'plain'], default='plain')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--n_cycles', type=int, default=4)
parser.add_argument('--n_samples_per_cycle', type=int, default=3)
parser.add_argument('--initial_lr', type=float, default=0.1)
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

path = './pretrained_models'
OOD_TRAINING = args.method in ['noneclass']
model_suffix = f'_{args.method}'

batch_size = 128
train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False)
ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(test_loader.dataset))

num_classes = 100 if args.dataset == 'CIFAR100' else 10
data_size = len(train_loader.dataset)

if OOD_TRAINING:
    # Doubles the amount of data
    batch_size *= 2
    data_size *= 2
    num_classes += 1

model = WideResNet(16, 4, num_classes, dropRate=0)
model.cuda()
model.train()
print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

num_batch = data_size/batch_size + 1
epoch_per_cycle = args.n_epochs // args.n_cycles
pbar = trange(args.n_epochs)
total_iters = args.n_epochs * num_batch
weight_decay = 5e-4

trainer = CSGHMCTrainer(
    model, args.n_cycles, args.n_samples_per_cycle, args.n_epochs, args.initial_lr,
    num_batch, total_iters, data_size, weight_decay
)
samples = []

for epoch in pbar:
    train_loss = 0
    if OOD_TRAINING:
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    for batch_idx, ((x_in, y_in), (x_out, _)) in enumerate(zip(train_loader, ood_loader)):
        trainer.model.train()
        trainer.model.zero_grad()

        m = len(x_in)  # Current batch size
        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)

        if OOD_TRAINING:
            x_out = x_out[:m].cuda(non_blocking=True)  # To make the length the same as x_in
            x = torch.cat([x_in, x_out], dim=0)

            # Noneclass: set x_out's target to be the last, additional, class
            # Last class, zero-indexed
            y_out = (num_classes-1) * torch.ones(m, device='cuda').long()
            y = torch.cat([y_in, y_out])
        else:
            x = x_in; y = y_in

        out = trainer.model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # The meat of the CSGMCMC method
        lr = trainer.adjust_lr(epoch, batch_idx)
        trainer.update_params(epoch)

        train_loss = 0.9*train_loss + 0.1*loss.item()

    # Save the last n_samples_per_cycle iterates of a cycle
    if (epoch % epoch_per_cycle) + 1 > epoch_per_cycle - args.n_samples_per_cycle:
        samples.append(copy.deepcopy(trainer.model.state_dict()))

    model.eval()
    pred = evalutil.predict(test_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )


if not os.path.exists(path):
    os.makedirs(path)

save_name = f'{path}/{args.dataset}{model_suffix}_csghmc.pt'
print(save_name)
torch.save(samples, save_name)

## Try loading and testing
samples_state_dicts = torch.load(save_name)
models = []

for state_dict in samples_state_dicts:
    _model = WideResNet(16, 4, num_classes, dropRate=0)
    _model.load_state_dict(state_dict)
    models.append(_model.cuda().eval())

print()

py_in = evalutil.predict_ensemble(test_loader, models).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')

