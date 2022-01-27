import torch
from torch.nn import functional as F
from torch import distributions as dists
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle as skshuffle
from torch.distributions.multivariate_normal import MultivariateNormal
from util import misc


@torch.no_grad()
def predict(dataloader, model, n_samples=1, T=1, delta=1, return_targets=False):
    training = model.training

    if training:
        model.eval()

    py = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for _ in range(n_samples):
            f_s = model.forward(x)
            
            if type(f_s) == tuple:
                f_s = f_s[0]

            py_ += torch.softmax(f_s/T, 1)
        py_ /= n_samples

        py.append(py_)
        targets.append(y)

    if training:
        model.train()

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_energy(dataloader, model, n_samples=1, T=1, delta=1, return_targets=False):
    training = model.training

    if training:
        model.eval()

    py = []
    energy = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        f_s = model.forward(x)
        py_ = torch.softmax(f_s/T, 1)
        energy_ = get_energy_score(f_s, T)

        py.append(py_)
        energy.append(energy_)
        targets.append(y)

    if training:
        model.train()

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(energy, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0), torch.cat(energy, dim=0)


@torch.no_grad()
def predict_laplace(test_loader, model, n_samples=10, apply_softmax=True, return_targets=False, delta=1):
    py = []
    targets = []

    for x, y in test_loader:
        x, y = delta*x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_llvb(dataloader, model, n_samples=1):
    training = model.training

    if training:
        model.eval()

    py = []
    for x, y in dataloader:
        x = x.cuda()

        outs = model(x, n_samples)  # n_samples x batch_size x n_classes

        py_ = 0
        for s in range(n_samples):
            py_ += 1/n_samples * torch.softmax(outs[s], 1)

        py.append(py_)

    if training:
        model.train()

    return torch.cat(py, dim=0)



@torch.no_grad()
def predict_ensemble(dataloader, models, T=1, delta=1, return_targets=False):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for model in models:
            f_s = model.forward(x)
            py_ += 1/len(models) * torch.softmax(f_s/T, 1)

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


def get_acc(py, target):
    return np.mean(np.argmax(py, 1) == target).mean()*100


def get_confidence(py, noneclass=False):
    # Probabilities of the "none" class---for OOD, higher is better
    # so, we need to "flip" it to make lower is better, just like non-noneclass models
    return py.max(-1) if not noneclass else py[:, :-1].max(-1)


def get_energy_score(logits, T=1):
    return -T*torch.logsumexp(logits/T, dim=1)


def get_mmc(py, noneclass=False):
    return get_confidence(py, noneclass).mean()*100


def get_auroc(py_in, py_out, noneclass_probs=False):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1

    if not noneclass_probs:
        examples = np.concatenate([py_in.max(1), py_out.max(1)])
    else:
        # Probabilities of the "none" class---for OOD, higher is better
        # so, we need to "flip" it to make lower is better, just like non-noneclass models
        examples = np.concatenate([1-py_in[:, -1], 1-py_out[:, -1]])

    return roc_auc_score(labels, examples)*100  # In percent


def get_aupr(py_in, py_out, noneclass_probs=False):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1

    if not noneclass_probs:
        examples = np.concatenate([py_in.max(1), py_out.max(1)])
    else:
        # Probabilities of the "none" class---for OOD, higher is better
        # so, we need to "flip" it to make lower is better, just like non-noneclass models
        examples = np.concatenate([1-py_in[:, -1], 1-py_out[:, -1]])

    prec, rec, thresh = precision_recall_curve(labels, examples)
    aupr = auc(rec, prec)
    return aupr.item()*100  # In percent


def get_fpr95(py_in, py_out, noneclass_probs=False):
    if not noneclass_probs:
        conf_in, conf_out = py_in.max(1), py_out.max(1)
    else:
        # Probabilities of the "none" class---for OOD, higher is better
        # so, we need to "flip" it to make lower is better, just like non-noneclass models
        conf_in, conf_out = 1-py_in[:, -1], 1-py_out[:, -1]

    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc)/len(conf_out)
    return fpr.item()*100, perc.item()*100


def get_calib(pys, y_true, M=15):
    # Put the confidence into M bins
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    # In percent
    ECE, MCE = ECE*100, MCE*100

    return ECE, MCE


def get_brier(py, target, n_classes=10):
    preds = torch.tensor(py).float()
    target_onehot = torch.tensor(misc.get_one_hot(target, n_classes)).float()
    return F.mse_loss(preds, target_onehot).item()


def get_loglik(py, target):
    # Sum since the test loglik is `\prod_i \log p(y=target_i | py_i)`
    probs = torch.tensor(py).float()
    targets = torch.tensor(target).long()
    return dists.Categorical(probs=probs).log_prob(targets).sum().item()


def timing(fun):
    """
    Return the original output(s) and a wall-clock timing in second.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    ret = fun()
    end.record()

    torch.cuda.synchronize()

    return ret, start.elapsed_time(end)/1000
