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
from scipy import stats
import krippendorff
from joblib import Parallel, delayed
import time
import pickle

def get_logs(log_dirs,num_iter=101,out_filename_surfixs=['_mcmean_k0_s0'],step_size=1):
    all_log_dirs_dict=dict()
    for log_dir in log_dirs:
        all_score_dict = dict()
        for out_filename_surfix in out_filename_surfixs:
            all_score = []
            for i in range(0,num_iter,step_size):
                cur_log_dir = os.path.join(log_dir,f'{i}')
                if i == 0:
                    cur_gt_dir = os.path.join(cur_log_dir,'out_gt.npy')
                    cur_pred_dir = os.path.join(cur_log_dir,'out_pred.npy')
                    if not os.path.exists(cur_pred_dir):
                        cur_gt_dir = os.path.join(cur_log_dir,f'out_gt{out_filename_surfix}.npy')
                        cur_pred_dir = os.path.join(cur_log_dir,f'out_pred{out_filename_surfix}.npy')
                    if not os.path.exists(cur_pred_dir):
                        cur_out_filename_surfix = out_filename_surfix.replace('noise','cmean')
                        cur_gt_dir = os.path.join(cur_log_dir,f'out_gt{cur_out_filename_surfix}.npy')
                        cur_pred_dir = os.path.join(cur_log_dir,f'out_pred{cur_out_filename_surfix}.npy')
                else:
                    cur_gt_dir = os.path.join(cur_log_dir,f'out_gt{out_filename_surfix}.npy')
                    cur_pred_dir = os.path.join(cur_log_dir,f'out_pred{out_filename_surfix}.npy')
                cur_pred = np.load(cur_pred_dir).astype(np.float32)
        #         print(cur_pred[0])
                cur_gt = np.load(cur_gt_dir)
                cur_values = np.expand_dims(cur_pred[list(range(len(cur_pred))),cur_gt],-1)
                all_score.append(cur_values)
                if i==0 and all_score_dict.get('mask',None) is None:
                    mask = (cur_pred.argmax(-1)==cur_gt)
                    all_score_dict['mask'] = mask
        #             mask = (cur_pred.max(-1)>np.median(cur_pred.max(-1)))
            all_score = np.stack(all_score,0)
            all_score_dict[out_filename_surfix] = all_score
        all_log_dirs_dict[log_dir] = all_score_dict
    return all_log_dirs_dict

def get_auc_preload(all_log_dirs_dict,log_dir,num_iter=101,mask_correct=True,use_class=False,use_gaussian=True,class_num=None,subset_inds = None,normalize=True,beta_id=2,out_filename_surfix=''):
    all_score = []
#     loss = torch.nn.CrossEntropyLoss(reduction='none')
    all_score = all_log_dirs_dict[log_dir][out_filename_surfix]
    all_score = np.stack(all_score,0)
    mask = all_log_dirs_dict[log_dir]['mask']
    if DEBUG:
        
        all_score = all_score[1:,...]
#         print(all_score.shape)
    
    if subset_inds is not None:
        all_score = all_score[:,subset_inds,:]
        mask = mask[subset_inds]
#     print(all_score.shape)
    if mask_correct:
        all_score = all_score[:,mask,:]
    
#     print(all_score.shape)
    if use_gaussian:
        all_score = gaussian_filter1d(all_score,sigma=1,axis=0)
    base_score = all_score[0,:]
    beta_3 = np.mean(all_score*base_score,1)/np.mean(base_score*base_score)
    beta_1 = np.mean(all_score,1)/np.mean(base_score)
    beta_2 = np.mean((all_score)/(base_score),1)
    
#     beta_3 = np.mean(np.abs(all_score*base_score),1)/np.mean(np.abs(base_score*base_score))
#     beta_1 = np.mean(np.abs(all_score),1)/np.mean(np.abs(base_score))
#     beta_2 = np.mean(np.abs(all_score)/np.abs(base_score),1)
    if beta_id == 1:
        beta = beta_1
    elif beta_id == 2:
        beta = beta_2
    else:
        beta = beta_3
    if normalize:
        all_score = all_score*np.expand_dims(beta,-1)
#     print('beta:',beta.shape)
    
#     print(all_score.shape,all_score[:,0])
    all_score = (all_score[0,:]-all_score[:-1,:])
#     print(all_score,all_score.shape)
#     beta = np.mean(ks*spdt_diff*exp_sum) / np.mean(all_score[0,:]*all_score[0,:])
#     exp_sum *= betas
    
    all_score = all_score.mean(-1)
    all_score = np.sum(all_score,axis=0)/(num_iter+1)
#     print(all_score.shape)
    
        
#     print(all_score.shape)
    return all_score#np.sum((all_diff[:-1,:]-all_diff[1:,:]),axis=0)#.mean(0)#all_score

def gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter = 101,experiment_list=None,random=None,use_gaussian=False,out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s3'],step_size=1,baseline=None):
    if experiment_list is None:
        # smooth, correct, beta
        experiment_list = [[False,False,False],
                           [True,True,True],
                          ]
    log_dict_smooth = get_logs(log_dirs_smooth,num_iter=n_iter,out_filename_surfixs=[out_filename_surfixs[1]],step_size=step_size)
    log_dict_h_edge = get_logs(log_dirs_h_edge,num_iter=n_iter,out_filename_surfixs=[out_filename_surfixs[0]],step_size=step_size)
            
           
    alphas = list()
    # corrs = list()
    # mean_scores_list = list()
    vars_list = list()
    for (smooth,mask_correct,normalize) in tqdm(experiment_list):
        beta_alpha = list()
        # beta_corr  = list()
        # beta_mean_scores = list()
        beta_var = list()
        beta_list = [2]
        for beta_id in beta_list:
            if smooth:
                out_filename_surfix = out_filename_surfixs[1]
                log_dict = log_dict_smooth
                log_dirs = log_dirs_smooth
            else:
                out_filename_surfix = out_filename_surfixs[0]
                log_dict = log_dict_h_edge
                log_dirs = log_dirs_h_edge
#             log_dict = get_logs(log_dirs,num_iter=n_iter,out_filename_surfixs=[out_filename_surfix],step_size=step_size)
            scores = np.stack([get_auc_preload(log_dict,l,mask_correct=mask_correct,use_class=True,use_gaussian=use_gaussian,class_num=None,normalize=normalize,beta_id=beta_id,out_filename_surfix=out_filename_surfix) for l in log_dirs],0)
            num_sample = 10000 if 'ImageNet' not in log_dirs[0] else 5040
            k_alphas = Parallel(n_jobs=N_JOBS,backend='threading',verbose=1)(delayed(get_mean_scores_for_all_logs)(log_dict,log_dirs,n_iter,mask_correct,True,False,None,np.random.choice(num_sample,num_sample),normalize,beta_id,out_filename_surfix) for i in range(10000))
            k_alphas = np.stack(k_alphas,axis=-1)
            k_alpha = k_alphas.mean(-1)
            k_alpha_var = k_alphas.var(-1)
            # mean_scores = scores.mean(-1)
            
            # corr = stats.spearmanr(scores.T).correlation
#             corr = corr[corr<1]
            beta_alpha.append(k_alpha)
            # beta_mean_scores.append(mean_scores)
            # beta_corr.append(corr)
            beta_var.append(k_alpha_var)
            # print(corr.shape)
        alphas.append(np.stack(beta_alpha,0))
        # mean_scores_list.append(np.stack(beta_mean_scores,0))
        # corrs.append(np.stack(beta_corr,0))
        vars_list.append(np.stack(beta_var,0))

    
    
    return alphas,vars_list

def get_mean_scores_for_all_logs(all_log_dirs_dict,log_dirs,num_iter=101,mask_correct=True,use_class=False,use_gaussian=True,class_num=None,subset_inds = None,normalize=True,beta_id=2,out_filename_surfix='',baseline=None):
    scores = np.stack([get_auc_preload(all_log_dirs_dict,l,mask_correct=mask_correct,use_class=True,use_gaussian=False,class_num=None,normalize=normalize,beta_id=beta_id,subset_inds=subset_inds,out_filename_surfix=out_filename_surfix) for l in log_dirs],0)
    start = time.time()
    
    return scores.mean(-1)

def get_global_consis_MNIST(log_dir,n_iter=101,ext='LoRF'):
    log_dirs_h_edge = [
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100',
        #  '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_KernelShap_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_DeepLift_no_iter_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Occlusion_no_iter_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Lime_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100',
        f'{log_dir}/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_h_edge_100'
        ]
    log_dirs_smooth = [
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_Saliency_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100',
        #  '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_KernelShap_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_DeepLift_no_iter_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Occlusion_no_iter_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Lime_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_GradientShap_no_iter_test_100',
        f'{log_dir}/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100',
        #     '/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_100'
        ]
    offset = 0
    if DEBUG:
        offset = 1
    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=n_iter+offset,experiment_list=None,out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,baseline=None)
    ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
    exp_id = 0
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'MNIST_global_consis_base_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()
    exp_id = 1
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'MNIST_global_consis_propose_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()

def get_global_consis_ImageNet(input_dir,n_iter=192,ext='LoRF'):
    log_dirs_h_edge = [
        f'{input_dir}/ResNet50_E_ImageNet_InputXGradient_val_h_edge_100',
        f'{input_dir}/ResNet50_E_ImageNet_Saliency_val_h_edge_100',
        f'{input_dir}/ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100',
        f'{input_dir}/ResNet50_E_ImageNet_GradientShap_val_h_edge_100',
        f'{input_dir}/ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100',
        f'{input_dir}/ResNet50_E_ImageNet_Deconvolution_val_h_edge_100',
            ]
    log_dirs_smooth = [
        f'{input_dir}/ResNet50_E_ImageNet_InputXGradient_val_100',
        f'{input_dir}/ResNet50_E_ImageNet_Saliency_val_100',
        f'{input_dir}/ResNet50_E_ImageNet_IntegratedGradients_val_100',
        f'{input_dir}/ResNet50_E_ImageNet_GradientShap_val_100',
        f'{input_dir}/ResNet50_E_ImageNet_GuidedBackprop_val_100',
        f'{input_dir}/ResNet50_E_ImageNet_Deconvolution_val_100',
            ]

    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=n_iter,experiment_list=None,out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,baseline=None)
    ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
    exp_id = 0
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'ImageNet_global_consis_base_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()
    exp_id = 1
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'ImageNet_global_consis_propose_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()

def get_global_consis_CIFAR10(input_dir,n_iter=454,ext='LoRF'):
    log_dirs_h_edge = [
        f'{input_dir}/vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100',
        f'{input_dir}/vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100',
        f'{input_dir}/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100',
        f'{input_dir}/vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100',
        f'{input_dir}/vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100',
        f'{input_dir}/vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100',
            ]
    log_dirs_smooth = [
        f'{input_dir}/vgg16_E_CIFAR10_InputXGradient_no_iter_test_100',
        f'{input_dir}/vgg16_E_CIFAR10_Saliency_no_iter_test_100',
        f'{input_dir}/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100',
        f'{input_dir}/vgg16_E_CIFAR10_GradientShap_no_iter_test_100',
        f'{input_dir}/vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100',
        f'{input_dir}/vgg16_E_CIFAR10_Deconvolution_no_iter_test_100',
            ]

    offset = 0
    if DEBUG:
        offset = 1
    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=n_iter-offset,experiment_list=None,out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,baseline=None)
    ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
    exp_id = 0
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'CIFAR10_global_consis_base_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()
    exp_id = 1
    label_list = ['InputXGradient','Saliency','IntegratedGradients','GuidedBackprop','GradientShap','Deconvolution']
    for lower,upper,y in zip(ci_result[0][exp_id][0],ci_result[1][exp_id][0],[0,1,2,3,4,5]):
        plt.plot((lower,upper),(y,y),'ro-',color='gray')
        text_y = y+0.2 if y<5 else y-0.3
        plt.text(x=lower,y=text_y,s=f'{upper-lower:.3f}')
    plt.xlabel(f'Mean AOPC score,L={n_iter}')
    plt.grid()
    plt.yticks(range(len([0,1,2,3,4,5])),label_list)
    plt.savefig(f'CIFAR10_global_consis_propose_{ext}_{n_iter}.eps', format='eps',bbox_inches='tight')
    plt.clf()

if __name__=='__main__':
    DEBUG = False
    N_JOBS = 24
    get_global_consis_MNIST('/home/yimupan/ECCV2022/SaliencyAnalysis/log_LeRF2',n_iter=100,ext='LeRF')
    get_global_consis_CIFAR10('/home/yimupan/ECCV2022/SaliencyAnalysis/log_LeRF2',n_iter=100,ext='LeRF')
    get_global_consis_MNIST('/home/yimupan/ECCV2022/SaliencyAnalysis/log_LeRF2',n_iter=212,ext='LeRF')
    get_global_consis_CIFAR10('/home/yimupan/ECCV2022/SaliencyAnalysis/log_LeRF2',n_iter=454,ext='LeRF')
    
    DEBUG = False
    get_global_consis_MNIST('/home/yimupan/ECCV2022/SaliencyAnalysis/log_MoRF_1',n_iter=100,ext='MoRF')
    get_global_consis_CIFAR10('/home/yimupan/ECCV2022/SaliencyAnalysis/log_MoRF_1',n_iter=100,ext='MoRF')
    get_global_consis_MNIST('/home/yimupan/ECCV2022/SaliencyAnalysis/log_MoRF_1',n_iter=212,ext='MoRF')
    get_global_consis_CIFAR10('/home/yimupan/ECCV2022/SaliencyAnalysis/log_MoRF_1',n_iter=454,ext='MoRF')