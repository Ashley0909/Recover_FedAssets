import random
from typing import Callable, Optional
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10
import os 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class MIXEDPoisonMNIST(MNIST):

    def __init__(
        self,
        root: str,
        benign_ratio: float,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        poisoning_rate: float = 0.4,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1
        self.train = train
        self.poisoning_rate = poisoning_rate

        self.trigger_handler = TriggerHandler("./triggers/trigger_white.png", 5, 9, self.width, self.height)
        indices = range(len(self.targets))
        """Let's say we only poison the last 40% of the dataset the malicious clients are holding"""
        malicious_ratio = 1 - benign_ratio if train else 1.0
        k = int(len(indices) * (1 - (self.poisoning_rate * malicious_ratio)))
        self.poi_indices = list(range(k,len(indices)))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            img = self.trigger_handler.put_trigger(img)
            if self.train:
                target = self.trigger_handler.trigger_label   # single target attack
                # if target == 9:                             # all-to-all attack
                #     target = 0
                # else:
                #     target = target + 1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class MIXEDPoisonCIFAR(CIFAR10):

    def __init__(
        self,
        root: str,
        benign_ratio: float,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        poisoning_rate: float = 0.4,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.train = train
        self.poisoning_rate = poisoning_rate

        self.trigger_handler = TriggerHandler("./triggers/trigger_white.png", 5, 9, self.width, self.height)
        indices = range(len(self.targets))
        """Let's say we only poison the last 40% of the dataset the malicious clients are holding"""
        malicious_ratio = 1 - benign_ratio if train else 1.0
        k = int(len(indices) * (1 - (self.poisoning_rate * malicious_ratio)))
        self.poi_indices = list(range(k,len(indices)))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            img = self.trigger_handler.put_trigger(img)
            if self.train:
                target = self.trigger_handler.trigger_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
def build_poisoned_training_set(tr, data_path, benign_ratio, dataset, poisoning_rate):
    if dataset == 'mnist':
        trainset = MIXEDPoisonMNIST(data_path, benign_ratio=benign_ratio, train=True, download=True, transform=tr, poisoning_rate=poisoning_rate)
    elif dataset == 'cifar10':
        trainset = MIXEDPoisonCIFAR(data_path, benign_ratio=benign_ratio, train=True, download=True, transform=tr, poisoning_rate=poisoning_rate)

    return trainset

def build_testset(tr, data_path, benign_ratio, dataset, poisoning_rate):
    if dataset == 'mnist':
        testset = MIXEDPoisonMNIST(data_path, benign_ratio=benign_ratio, train=False, download=True, transform=tr, poisoning_rate=poisoning_rate)
    elif dataset == 'cifar10':
        testset = MIXEDPoisonCIFAR(data_path, benign_ratio=benign_ratio, train=False, download=True, transform=tr, poisoning_rate=poisoning_rate)

    return testset