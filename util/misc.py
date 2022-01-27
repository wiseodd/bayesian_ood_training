import numpy as np
import torch
import torch.nn.functional as F
import math


def dirichlet_nll_ood(logits, prec=1, num_classes=10):
    alpha = prec*F.log_softmax(logits, 1).exp()
    return torch.sum(alpha*math.log(num_classes) + torch.lgamma(alpha), -1)


def dirichlet_nll(logits, targets, prec=1, num_classes=10):
    alpha = prec*F.log_softmax(logits, 1).exp()
    # alpha = F.softplus(logits) + 1e-1
    print(alpha)
    return -torch.sum((alpha-1)*torch.log(targets+1e-8)) - torch.lgamma(alpha.sum(-1)) + torch.lgamma(alpha).sum(-1)


def ood_nll(y_pred_in, y_true_in, y_pred_out, prec=1, num_classes=10):
    nll_in = F.cross_entropy(y_pred_in, y_pred_out, reduction='sum')
    nll_out = dirichlet_nll(y_pred_out, prec, num_classes)
    return nll_in + 1/num_classes * nll_out


def get_one_hot(targets, nb_classes, torch=False):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    onehot = res.reshape(list(targets.shape)+[nb_classes])

    if torch:
        onehot = torch.tensor(onehot).float()

    return onehot


def label_smoothing(x_onehot, eps=1e-8):
    N, K = x_onehot.shape
    labels = x_onehot.argmax(-1)
    x_smooth = eps * torch.ones_like(x_onehot, device='cuda')
    x_smooth[range(N), labels] = 1 - (K-1)*eps
    return x_smooth
