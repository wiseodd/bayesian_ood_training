##########################################################################
#
#  Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
#
##########################################################################

import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import numpy as np
import scipy.ndimage.filters as filters
import util.preproc as pre

from bisect import bisect_left
import os

from PIL import Image
import pickle


torch.multiprocessing.set_sharing_strategy('file_system')

train_batch_size = 128
test_batch_size = 128

path = os.path.expanduser('~/Datasets')


def MNIST(train=True, batch_size=None, augm_flag=True, val_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.MNIST(path, train=train, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


def EMNIST(train=False, batch_size=None, augm_flag=False, val_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor(), pre.Transpose()] #EMNIST is rotated 90 degrees from MNIST
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.EMNIST(path, split='letters',
                              train=train, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=1)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size
        dataset_val, dataset_test = data_utils.random_split(dataset, (val_size, test_size))
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=1)
        return val_loader, test_loader


def KMNIST(train=True, batch_size=None, augm_flag=True, val_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.KMNIST(path, train=train, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


def FMNIST(train=False, batch_size=None, augm_flag=False, val_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.FashionMNIST(path, train=train, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=1)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size
        dataset_val, dataset_test = data_utils.random_split(dataset, (val_size, test_size))
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=1)
        return val_loader, test_loader


def FMNIST3D(train=False, batch_size=None, augm_flag=False, val_size=None):
    transform_base = [transforms.Resize([32, 32]), transforms.Grayscale(3), transforms.ToTensor()]
    transform = transforms.Compose(transform_base)

    dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    return loader


def RotatedMNIST(train=True, batch_size=None, augm_flag=True, val_size=None, angle=60):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose([transforms.RandomRotation((angle,angle))] + transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.MNIST(path, train=train, transform=transform)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


def GrayCIFAR10(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            pre.Gray()
                       ])]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR10(path, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def Noise(dataset, train=True, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        pre.PermutationNoise(),
        pre.GaussianFilter(),
        pre.ContrastRescaling()
    ])
    if dataset=='MNIST':
        dataset = datasets.MNIST(path, train=train, transform=transform)
    elif dataset=='FMNIST':
        dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    elif dataset=='SVHN':
        dataset = datasets.SVHN(path, split='train' if train else 'test', transform=transform)
    elif dataset=='CIFAR10':
        dataset = datasets.CIFAR10(path, train=train, transform=transform)
    elif dataset=='CIFAR100':
        dataset = datasets.CIFAR100(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    loader = PrecomputeLoader(loader, batch_size=batch_size, shuffle=True)
    return loader


def UniformNoise(dataset, train=False, size=2000, batch_size=None):
    if batch_size == None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    if dataset in ['MNIST', 'FMNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        shape = (3, 32, 32)

    data = torch.rand((size,) + shape)
    train = torch.utils.data.TensorDataset(data, torch.zeros_like(data))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def CIFAR10(train=True, batch_size=None, augm_flag=True, val_size=None, mean=None, std=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_base += [transforms.Normalize(mean, std)]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR10(path, train=train, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


def CIFAR100(train=False, batch_size=None, augm_flag=True, val_size=None, mean=None, std=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_base += [transforms.Normalize(mean, std)]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR100(path, train=train, transform=transform)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


def SVHN(train=True, batch_size=None, augm_flag=True, val_size=None, mean=None, std=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    if train:
        split = 'train'
    else:
        split = 'test'

    transform_base = [transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_base += [transforms.Normalize(mean, std)]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.SVHN(path, split=split, transform=transform, download=True)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=train, num_workers=4)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size

        # The split is fixed, since the seed is also fixed
        dataset_val, dataset_test = data_utils.random_split(
            dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                shuffle=train, num_workers=4)
        return val_loader, test_loader


# LSUN classroom
def LSUN_CR(train=False, batch_size=None, augm_flag=False, mean=None, std=None):
    if train:
        print('Warning: Training set for LSUN not available')
    if batch_size is None:
        batch_size=test_batch_size

    transform_base = [transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_base += [transforms.Normalize(mean, std)]

    transform = transforms.Compose([
            transforms.Resize(size=(32, 32))
        ] + transform_base)
    data_dir = path
    dataset = datasets.LSUN(data_dir, classes=['classroom_val'], transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    return loader


def PrecomputeLoader(loader, batch_size=100, shuffle=True):
    X = []
    L = []
    for x,l in loader:
        X.append(x)
        L.append(l)
    X = torch.cat(X, 0)
    L = torch.cat(L, 0)

    train = data_utils.TensorDataset(X, L)
    return data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)


def CorruptedCIFAR10(distortion, severity=1, batch_size=None, meanstd=None):
    if batch_size==None:
        batch_size=test_batch_size

    transform_base = [transforms.ToTensor()]

    if meanstd is not None:
        transform_base.append(transforms.Normalize(*meanstd))

    transform = transforms.Compose(transform_base)
    dataset = CorruptedCIFAR10Dataset(path, distortion, severity, transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True)
    return loader


class CorruptedCIFAR10Dataset(datasets.VisionDataset):
    """ https://arxiv.org/abs/1903.12261 """

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    base_folder = 'CIFAR-10-C'

    def __init__(self, root, distortion, severity=1, transform=None):
        assert distortion in self.distortions
        assert severity >= 1 and severity <= 5

        super(CorruptedCIFAR10Dataset, self).__init__(root, transform=transform, target_transform=None)

        file_name = f'{distortion}.npy'
        start_idx = (severity-1)*10000
        end_idx = start_idx + 10000  # 10000 is the test set size of CIFAR-10

        self.data = np.load(os.path.join(self.root, self.base_folder, file_name))
        self.data = self.data[start_idx:end_idx]  # NHWC
        self.targets = np.load(os.path.join(self.root, self.base_folder, 'labels.npy'))
        self.targets = self.targets[start_idx:end_idx]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def ImageNet32(train=True, dataset='CIFAR10', batch_size=None, augm_flag=True, val_size=None, meanstd=None):
    assert dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    if dataset in ['MNIST', 'FMNIST']:
        img_size = 28
        transform_base = [
            transforms.Resize(img_size),
            transforms.Grayscale(1),  # Single-channel grayscale image
            transforms.ToTensor()
        ]

        if meanstd is not None:
            transform_base.append(transforms.Normalize(*meanstd))

        transform_train = transforms.Compose(
            [transforms.RandomCrop(28, padding=2)] + transform_base
        )
    else:
        img_size = 32
        transform_base = [transforms.ToTensor()]

        if meanstd is not None:
            transform_base.append(transforms.Normalize(*meanstd))

        padding_mode = 'edge' if dataset == 'SVHN' else 'reflect'
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4, padding_mode=padding_mode),
            ] + transform_base
        )

    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    dataset = ImageNet32Dataset(path, train=train, transform=transform)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=train, num_workers=4, pin_memory=True)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size
        dataset_val, dataset_test = data_utils.random_split(dataset, (val_size, test_size))
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                 shuffle=train, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                 shuffle=train, num_workers=4, pin_memory=True)
        return val_loader, test_loader


class ImageNet32Dataset(datasets.VisionDataset):
    """ https://arxiv.org/abs/1707.08819 """

    base_folder = 'Imagenet32'
    train_list = [f'train_data_batch_{i}' for i in range(1, 11)]
    test_list = ['val_data']

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(ImageNet32Dataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.offset = 0  # offset index---for inducing randomness

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        # Shift the index by an offset, which can be chosen randomly
        index = (index + self.offset) % len(self)

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


# ==================================================================================================


datasets_dict = {
    'MNIST': MNIST,
    'KMNIST': KMNIST,
    'FMNIST': FMNIST,
    'GrayCIFAR10': GrayCIFAR10,
    'EMNIST': EMNIST,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'SVHN': SVHN,
    'LSUN': LSUN_CR,
    'FMNIST3D': FMNIST3D,
    'UniformNoise': UniformNoise,
    'Noise': Noise,
 }
