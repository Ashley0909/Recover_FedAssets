from omegaconf import DictConfig
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler

from mixedbackdoor import build_poisoned_training_set, build_testset
from dataset_preparation import _partition_data
import matplotlib.pyplot as plt
import numpy as np


def get_mnist(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def get_cifar(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_clientdataset(config: DictConfig,
                    num_partitions: int, 
                    batch_size: int,
                    dataset: str,
                    val_ratio: float = 0.1,
                    ):
    
    """Import mixed poisoned dataset"""
    if dataset == 'mnist':
        tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])
    elif dataset == 'cifar10':
        tr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    mixedtrain = build_poisoned_training_set(tr, data_path = './data', benign_ratio=config.ratio_benign_client, dataset=dataset, poisoning_rate=config.poisoning_rate)

    testset = build_testset(tr, data_path = './data', benign_ratio=config.ratio_benign_client, dataset=dataset, poisoning_rate=config.poisoning_rate)

    attacker_testset = build_testset(tr, data_path = './data', benign_ratio=config.ratio_benign_client, dataset=dataset, poisoning_rate=1.0)

    """Partition the data"""
    goodtrainsets, badtrainsets = _partition_data(
        mixedtrain,
        num_partitions,
        benign_ratio=config.ratio_benign_client,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        dirichlet=config.dirichlet,
        alpha=config.alpha,
        seed=42,
    )

    bdtrainloaders = []
    bdvalloaders = []
    cleantrainloaders = []
    cleanvalloaders = []

    for bdtrainset_ in badtrainsets:
        num_total = len(bdtrainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(bdtrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        bdtrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
        bdvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True))

    for ctrainset_ in goodtrainsets:
        num_total = len(ctrainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(ctrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        cleantrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
        cleanvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True))

    testloader = DataLoader(testset, batch_size=128)

    attack_testloader = DataLoader(attacker_testset, batch_size=128)

    return bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders, testloader, attack_testloader


def show_images_labels(images, labels, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(12,5))

    for i in range(num_samples):
        image = images[i]
        label = labels[i]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    plt.show(block=True)

