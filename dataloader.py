import os

import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from torchvision import transforms

import yaml
import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __followitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), img

    def __len__(self):
        return len(self.base_dataset)


################################################################""


def get_dataloaders(config, use_cuda):
    """
    Arguments :
        ratio       : float in ]0,1[   , pourcentage de donnees de train qui vont etre utilisees pour la validation
        batch_size  : integer          , taille des batchs
        num_workers : integer          , nombre de processus paralleles pour charger les donnees
        use_cuda    : Boolean          , Utilisation d'un gpu
        data_dir    :
    """

    ratio = config["data"]["valid_ratio"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    data_dir = config["data"]["trainpath"]

    if config["data"]["dataset"] in ["CIFAR10", "CIFAR100"]:
        dataset = eval(
            f"torchvision.datasets.{config['data']['dataset']}(root=data_dir, train=True, download=True)"
        )
    else:
        raise ValueError("Can only process CIFAR10 or CIFAR100")

    classes = dataset.classes
    num_classes = len(classes)

    train_size = int((1 - ratio) * len(dataset))
    valid_size = int(ratio * len(dataset))

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    # Creation d'un dataloader intermediaire afin de calculer la moyenne et l'ecart type de l'ensemble d'entrainement

    normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    normalizing_loader = torch.utils.data.DataLoader(
        dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Compute mean and variance from the training set
    mean, std = utils.mean_std(normalizing_loader)

    # Data Augmentation and Normalization
    train_augmentation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    test_normalization = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Application des transformations
    train_dataset = DatasetTransformer(train_dataset, train_augmentation)
    valid_dataset = DatasetTransformer(valid_dataset, test_normalization)

    a = train_dataset.__followitem__(0)

    # Creation dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    # test_loader  = torch.utils.data.DataLoader(test_dataset , batch_size=batch_size, shuffle=False, num_workers=num_workers)

    X, _ = train_dataset[0]
    input_size = tuple(X.shape)  # C, H, W

    return train_loader, valid_loader, input_size, num_classes, classes, a, mean, std


def test_loading_test():
    with open("\classification-cifar-100\Config.yaml", "r") as f:
        config = yaml.safe_load(f)
    (
        train_lod,
        valid_lod,
        inputsize,
        numclasses,
        classes,
        a,
        mean,
        std,
    ) = get_dataloaders(config, False)
    print(f"The loaded input size is {inputsize}, with {numclasses} classes")
    print(f"The classes are {classes}")


def test_show_image():
    # read yaml file
    with open("classification-cifar-100/Config.yaml", "r") as f:
        config = yaml.safe_load(f)
    (
        train_loader,
        valid_loader,
        input_size,
        num_classes,
        classes,
        a,
        mean,
        std,
    ) = get_dataloaders(config, True)
    imgs, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(20, 5), facecolor="w")
    for i in range(10):
        ax = plt.subplot(1, 10, i + 1)
        plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(classes[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("CIFAR100_samples.png", bbox_inches="tight")
    plt.show()


def show_tansformation():
    # read yaml file
    with open("classification-cifar-100/Config.yaml", "r") as f:
        config = yaml.safe_load(f)
    (
        train_loader,
        valid_loader,
        input_size,
        num_classes,
        classes,
        a,
        mean,
        std,
    ) = get_dataloaders(config, True)
    fig = plt.figure(figsize=(20, 5), facecolor="w")
    tran, img = a
    convert = transforms.ToTensor()
    img = convert(img)
    img = img.swapaxes(0, 1)  # probleme de dimension (32, 32, 3) au lieu de (3, 32, 32)
    img = img.swapaxes(1, 2)
    tran = tran.swapaxes(
        0, 1
    )  # probleme de dimension (32, 32, 3) au lieu de (3, 32, 32)
    tran = tran.swapaxes(1, 2)

    ax = plt.subplot(1, 2, 2)
    plt.imshow(img, vmin=0, vmax=1.0, cmap=cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(1, 2, 1)
    plt.imshow(tran, vmin=0, vmax=1.0, cmap=cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()


if __name__ == "__main__":
    test_loading_test()
    show_tansformation()
    print("done")
