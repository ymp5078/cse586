import torch
from tqdm import tqdm
from torchvision import transforms
from config import configs as config_lib
import os
import shutil
import json
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from data import datasets
from model import models as model_lib
from model import preprocessing as transform_lib
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from captum.attr import GuidedGradCam,IntegratedGradients,NoiseTunnel,GradientShap,GuidedBackprop,Saliency,InputXGradient,KernelShap,DeepLift,Deconvolution
from captum.attr import visualization as viz

def gen_MNIST(L = 212):
    img_id = 10
    
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    # plt.figure(figsize=( 60,20))
    fig, axs = plt.subplots(5, 7, sharex='all', sharey='all',figsize=(14, 10),gridspec_kw={'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35)
    map_files=[f'SimpleCNN_E_MNIST_{i}.npy' for i in range(L+2)]
    map_dirs_MoRF = [
    #     'SimpleCNN_DeepLiftShap_no_iter_test_100_saliency_maps',
        'SimpleCNN_InputXGradient_no_iter_test_100_saliency_maps_mean',
        'SimpleCNN_Saliency_no_iter_test_100_saliency_maps_mean',
        'SimpleCNN_IntegratedGradients_no_iter_test_100_saliency_maps_mean',
        
        'SimpleCNN_GradientShap_no_iter_test_100_saliency_maps_mean',
    #     'SimpleCNN_Occlusion_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_DeepLift_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_KernelShap_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_Lime_no_iter_test_100_saliency_maps',
        'SimpleCNN_GuidedBackprop_no_iter_test_100_saliency_maps_mean',
        'SimpleCNN_Deconvolution_no_iter_test_100_saliency_maps_mean',
    #     'SimpleCNN_FeatureAblation_no_iter_test_100_saliency_maps',
        
    ] 
    map_dirs_LeRF = [
    #     'SimpleCNN_DeepLiftShap_no_iter_test_100_saliency_maps',
        'SimpleCNN_InputXGradient_no_iter_test_100_saliency_maps',
        'SimpleCNN_Saliency_no_iter_test_100_saliency_maps',
        'SimpleCNN_IntegratedGradients_no_iter_test_100_saliency_maps',
        
        'SimpleCNN_GradientShap_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_Occlusion_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_DeepLift_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_KernelShap_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_Lime_no_iter_test_100_saliency_maps',
        'SimpleCNN_GuidedBackprop_no_iter_test_100_saliency_maps',
        'SimpleCNN_Deconvolution_no_iter_test_100_saliency_maps',
    #     'SimpleCNN_FeatureAblation_no_iter_test_100_saliency_maps',
        
    ] 
    original_dataset = datasets.E_MNIST('./data/MNIST',transform=transform, split='test')
    test_datasets_no_smooth_MoRF = [datasets.E_MNIST('./data/MNIST',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/MNIST/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]
    test_datasets_smooth_MoRF = [datasets.E_MNIST('./data/MNIST',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/MNIST/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]

    test_datasets_no_smooth_LeRF = [datasets.E_MNIST('./data/MNIST',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/MNIST/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]
    test_datasets_smooth_LeRF = [datasets.E_MNIST('./data/MNIST',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/MNIST/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]


    fontsize=10
    axs[0, 6].set_title('Original Image',fontsize=fontsize)
    axs[0, 0].set_ylabel('Saliency Map',fontsize=fontsize)
    axs[1, 0].set_ylabel(f'MoRF,L={L}',fontsize=fontsize)
    axs[2, 0].set_ylabel(f'MoRF,L={L},Smoothed',fontsize=fontsize)
    axs[3, 0].set_ylabel(f'LeRF,L={L}',fontsize=fontsize)
    axs[4, 0].set_ylabel(f'LeRF,L={L},Smoothed',fontsize=fontsize)
    for i in range(len(map_dirs_LeRF)):
        test_dataset_no_smooth = test_datasets_no_smooth_MoRF[i]
        test_dataset_smooth = test_datasets_smooth_MoRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        axs[0,i].imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='binary')
        axs[0,i].set_xticks(())
        axs[0,i].set_yticks(())
        ylb = map_dirs_LeRF[i].split('_')[1]
    #     axs[i, 0].set_xlabel('x',fontsize=20)

        axs[0,i].set_title(ylb,fontsize=fontsize)
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[1,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[2,i].imshow(img*0.5+0.5,cmap='gray')

        test_dataset_no_smooth = test_datasets_no_smooth_LeRF[i]
        test_dataset_smooth = test_datasets_smooth_LeRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[3,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[4,i].imshow(img*0.5+0.5,cmap='gray')
    img, lb = original_dataset[img_id]
    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    for i in range(5):
        axs[i,6].imshow(img*0.5+0.5,cmap='gray')
    # img_plt = plt.imshow(img*0.5+0.5)
    plt.tight_layout(pad=0.0)
    plt.savefig('MNIST_examples.eps',format='eps',bbox_inches='tight')
    # plt.show()

def gen_ImageNet(L=191):
    
    img_id = 10
    # L = 191
    transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.PILToTensor(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    # plt.figure(figsize=( 60,20))
    fig, axs = plt.subplots(5, 7, sharex='all', sharey='all',figsize=( 14,10),gridspec_kw={'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35)
    #data/ImageNet/ResNet50_GradientShap_val_h_edge_100_saliency_maps_MoRF/val/ResNet50_E_ImageNet_2
    map_files=[f'ResNet50_E_ImageNet_0_{i}.npy' for i in range(L)]
    map_dirs_MoRF = [
        'ResNet50_InputXGradient_val_h_edge_100_saliency_maps_MoRF',
        'ResNet50_Saliency_val_h_edge_100_saliency_maps_MoRF',
        'ResNet50_IntegratedGradients_val_h_edge_100_saliency_maps_MoRF',
        'ResNet50_GradientShap_val_h_edge_100_saliency_maps_MoRF',
        'ResNet50_GuidedBackprop_val_h_edge_100_saliency_maps_MoRF',
        'ResNet50_Deconvolution_val_h_edge_100_saliency_maps_MoRF',
        
    ] 
    map_dirs_LeRF = [
        'ResNet50_InputXGradient_val_h_edge_100_saliency_maps_LeRF',
        'ResNet50_Saliency_val_h_edge_100_saliency_maps_LeRF',
        'ResNet50_IntegratedGradients_val_h_edge_100_saliency_maps_LeRF',
        'ResNet50_GradientShap_val_h_edge_100_saliency_maps_LeRF',
        'ResNet50_GuidedBackprop_val_h_edge_100_saliency_maps_LeRF',
        'ResNet50_Deconvolution_val_h_edge_100_saliency_maps_LeRF',
    ] 

    test_datasets_no_smooth_MoRF = [datasets.E_ImageNet('./data/ImageNet',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/ImageNet/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]
    test_datasets_smooth_MoRF = [datasets.E_ImageNet('./data/ImageNet',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/ImageNet/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]

    test_datasets_no_smooth_LeRF = [datasets.E_ImageNet('./data/ImageNet',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/ImageNet/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]
    test_datasets_smooth_LeRF = [datasets.E_ImageNet('./data/ImageNet',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=1,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/ImageNet/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]


    fontsize=10
    axs[0, 0].set_ylabel('Saliency Map',fontsize=fontsize)
    axs[1, 0].set_ylabel(f'MoRF,L={L}',fontsize=fontsize)
    axs[2, 0].set_ylabel(f'MoRF,L={L},Smoothed',fontsize=fontsize)
    axs[3, 0].set_ylabel(f'LeRF,L={L}',fontsize=fontsize)
    axs[4, 0].set_ylabel(f'LeRF,L={L},Smoothed',fontsize=fontsize)
    for i in range(len(map_dirs_MoRF)):
        test_dataset_no_smooth = test_datasets_no_smooth_MoRF[i]
        test_dataset_smooth = test_datasets_smooth_MoRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        axs[0,i].imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='binary')
        axs[0,i].set_xticks(())
        axs[0,i].set_yticks(())
        ylb = map_dirs_LeRF[i].split('_')[1]
    #     axs[i, 0].set_xlabel('x',fontsize=20)

        axs[0,i].set_title(ylb,fontsize=fontsize)
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[1,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[2,i].imshow(img*0.5+0.5,cmap='gray')

        test_dataset_no_smooth = test_datasets_no_smooth_LeRF[i]
        test_dataset_smooth = test_datasets_smooth_LeRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        axs[0,i].imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='binary')
        axs[0,i].set_xticks(())
        axs[0,i].set_yticks(())
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[3,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[4,i].imshow(img*0.5+0.5,cmap='gray')
    img, lb = original_dataset[img_id]
    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    for i in range(5):
        axs[i,6].imshow(img*0.5+0.5,cmap='gray')
    
    # img_plt = plt.imshow(img*0.5+0.5)
    plt.tight_layout(pad=0.0)
    plt.savefig('ImageNet_examples.eps',format='eps',bbox_inches='tight')# plt.show()


def gen_CIFAR(L=454):
    
    img_id = 10
    # L = 200
    transform = transforms.Compose([
                                            transforms.PILToTensor(),
                                            transforms.ConvertImageDtype(torch.float),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    # plt.figure(figsize=( 60,20))
    fig, axs = plt.subplots(5, 7, sharex='all', sharey='all',figsize=( 14,10),gridspec_kw={'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35)
    map_files=[f'vgg16_E_CIFAR10_{i}.npy' for i in range(L)]
    map_dirs_MoRF = [
    #     'vgg16_DeepLiftShap_no_iter_test_100_saliency_maps',
        'vgg16_InputXGradient_no_iter_test_100_saliency_maps_mean',
        'vgg16_Saliency_no_iter_test_100_saliency_maps_mean',
        'vgg16_IntegratedGradients_no_iter_test_100_saliency_maps_mean',
        'vgg16_GradientShap_no_iter_test_100_saliency_maps_mean',
    #     'vgg16_KernelShap_no_iter_test_100_saliency_maps',
        'vgg16_GuidedBackprop_no_iter_test_100_saliency_maps_mean',
    #     'vgg16_DeepLift_no_iter_test_100_saliency_maps',
    #     'vgg16_LRP_no_iter_test_100_saliency_maps',
    #     'vgg16_Lime_no_iter_test_100_saliency_maps',
        'vgg16_Deconvolution_no_iter_test_100_saliency_maps_mean',
    ] 
    map_dirs_LeRF = [
    #     'vgg16_DeepLiftShap_no_iter_test_100_saliency_maps',
        'vgg16_InputXGradient_no_iter_test_100_saliency_maps',
        'vgg16_Saliency_no_iter_test_100_saliency_maps',
        'vgg16_IntegratedGradients_no_iter_test_100_saliency_maps',
        'vgg16_GradientShap_no_iter_test_100_saliency_maps',
    #     'vgg16_KernelShap_no_iter_test_100_saliency_maps',
        'vgg16_GuidedBackprop_no_iter_test_100_saliency_maps',
    #     'vgg16_DeepLift_no_iter_test_100_saliency_maps',
    #     'vgg16_LRP_no_iter_test_100_saliency_maps',
    #     'vgg16_Lime_no_iter_test_100_saliency_maps',
        'vgg16_Deconvolution_no_iter_test_100_saliency_maps',
    ] 
    original_dataset = datasets.E_CIFAR10('./data/CIFAR10',transform=transform, split='test')
    test_datasets_no_smooth_MoRF = [datasets.E_CIFAR10('./data/CIFAR10',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=0.5,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/CIFAR10/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]
    test_datasets_smooth_MoRF = [datasets.E_CIFAR10('./data/CIFAR10',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=0.5,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/CIFAR10/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_MoRF]

    test_datasets_no_smooth_LeRF = [datasets.E_CIFAR10('./data/CIFAR10',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=0.5,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/CIFAR10/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]
    test_datasets_smooth_LeRF = [datasets.E_CIFAR10('./data/CIFAR10',transform=transform, split='test',smooth_map=True,kernel_size=3,sigma=0.5,use_basemap=False,iter_map=False,map_files=map_files,map_dir=f'./data/CIFAR10/{map_dir}',download=True,map_out=True,pertub='cmean',cur_iter=L+1) for map_dir in map_dirs_LeRF]


    fontsize=10
    axs[0, 0].set_ylabel('Saliency Map',fontsize=fontsize)
    axs[1, 0].set_ylabel(f'MoRF,L={L}',fontsize=fontsize)
    axs[2, 0].set_ylabel(f'MoRF,L={L},Smoothed',fontsize=fontsize)
    axs[3, 0].set_ylabel(f'LeRF,L={L}',fontsize=fontsize)
    axs[4, 0].set_ylabel(f'LeRF,L={L},Smoothed',fontsize=fontsize)
    axs[0, 6].set_title('Original Image',fontsize=fontsize)
    for i in range(len(map_dirs_LeRF)):
        test_dataset_no_smooth = test_datasets_no_smooth_MoRF[i]
        test_dataset_smooth = test_datasets_smooth_MoRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        axs[0,i].imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='binary')
        axs[0,i].set_xticks(())
        axs[0,i].set_yticks(())
        ylb = map_dirs_LeRF[i].split('_')[1]
    #     axs[i, 0].set_xlabel('x',fontsize=20)

        axs[0,i].set_title(ylb,fontsize=fontsize)
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[1,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[2,i].imshow(img*0.5+0.5,cmap='gray')

        test_dataset_no_smooth = test_datasets_no_smooth_LeRF[i]
        test_dataset_smooth = test_datasets_smooth_LeRF[i]
        img, lb = test_dataset_no_smooth[img_id]
        base_map = test_dataset_no_smooth.base_map[img_id]
        axs[0,i].imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='binary')
        axs[0,i].set_xticks(())
        axs[0,i].set_yticks(())
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[3,i].imshow(img*0.5+0.5,cmap='gray')

        img, lb = test_dataset_smooth[img_id]
        base_map = test_dataset_smooth.base_map[img_id]
        img = img.cpu().detach().numpy().transpose((1, 2, 0))
        axs[4,i].imshow(img*0.5+0.5,cmap='gray')
    img, lb = original_dataset[img_id]
    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    for i in range(5):
        axs[i,6].imshow(img*0.5+0.5,cmap='gray')
        

    # img_plt = plt.imshow(img*0.5+0.5)
    plt.tight_layout(pad=0.0)
    plt.savefig('CIFAR_examples.eps',format='eps',bbox_inches='tight')
    # plt.show()

if __name__=='__main__':
    gen_MNIST()
    gen_CIFAR()