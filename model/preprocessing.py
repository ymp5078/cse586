import torch
from torchvision import transforms

class E_CIFAR10:
    train_transforms = transforms.Compose([
                                        transforms.PILToTensor(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])
                                    ])
    eval_transforms = transforms.Compose([
                                        transforms.PILToTensor(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])
                                    ])
class E_Food101:
    train_transforms = transforms.Compose([
                                        transforms.PILToTensor(),
                                        transforms.RandomResizedCrop(size=(224, 224),scale=(0.5, 1.0),),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize([0.561, 0.440, 0.312], [0.252, 0.256, 0.259])
                                    ])
    eval_transforms = transforms.Compose([
                                        transforms.PILToTensor(),
                                        transforms.Resize(size=(224,224)),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize([0.561, 0.440, 0.312], [0.252, 0.256, 0.259])
                                    ])
                                
class E_MNIST:
    train_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    eval_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])

class E_ImageNet:
    train_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                    ])
    eval_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                    ])

PREPROCESSING = {
    'E_CIFAR10':E_CIFAR10,
    'E_Food101':E_Food101,
    'E_MNIST':E_MNIST,
    'E_ImageNet':E_ImageNet
}

def get_preprocessing(dataset_name):
    return PREPROCESSING[dataset_name]