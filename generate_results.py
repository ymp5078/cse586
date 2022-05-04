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
from scipy.ndimage import gaussian_filter1d
import sys


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

def get_auc(log_dir,num_iter=101,mask_correct=True,use_class=False,use_gaussian=True,class_num=None,subset_inds = None,normalize=True,beta_id=2,out_filename_surfix='',step_size=1):
    all_score = []
#     loss = torch.nn.CrossEntropyLoss(reduction='none')
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
            if not os.path.exists(cur_pred_dir):
                cur_out_filename_surfix = out_filename_surfix.replace('s1','s0')
                cur_gt_dir = os.path.join(cur_log_dir,f'out_gt{cur_out_filename_surfix}.npy')
                cur_pred_dir = os.path.join(cur_log_dir,f'out_pred{cur_out_filename_surfix}.npy')
        else:
            cur_gt_dir = os.path.join(cur_log_dir,f'out_gt{out_filename_surfix}.npy')
            cur_pred_dir = os.path.join(cur_log_dir,f'out_pred{out_filename_surfix}.npy')
        cur_pred = np.load(cur_pred_dir).astype(np.float32)
#         print(cur_pred[0])
        cur_gt = np.load(cur_gt_dir)
        cur_values = torch.nn.functional.cross_entropy(torch.from_numpy(cur_pred),torch.from_numpy(cur_gt),reduction='none').numpy()
#         cur_pred = torch.nn.functional.softmax(torch.from_numpy(cur_pred),dim=-1).numpy()
#         cur_values = cur_pred
        cur_values = np.expand_dims(cur_values,-1)
        if use_class:
            cur_values = np.expand_dims(cur_pred[list(range(len(cur_pred))),cur_gt],-1)
#         print(cur_values.max(-1))
#         print('dis_cur:',cur_values.mean(),cur_values.var())
        if class_num is not None:
            cur_class = class_num == cur_gt
            cur_values = cur_values[cur_class]
            cur_gt = cur_gt[cur_class]
            cur_pred = cur_pred[cur_class,]
        all_score.append(cur_values)
        if i==0:
            mask = (cur_pred.argmax(-1)==cur_gt)
#             mask = (cur_pred.max(-1)>np.median(cur_pred.max(-1)))
    all_score = np.stack(all_score,0)
    if subset_inds is not None:
        all_score = all_score[:,subset_inds,:]
        mask = mask[subset_inds]
#     print(all_score.shape)
    if mask_correct:
        all_score = all_score[:,mask,:]
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
                           [False,True,False],
                           [False,False,True],
                           [False,True,True],
                           [True,False,False],
                           [True,True,False],
                           [True,False,True],
                           [True,True,True],
                          ]
    log_dict_smooth = get_logs(log_dirs_smooth,num_iter=n_iter,out_filename_surfixs=[out_filename_surfixs[1]],step_size=step_size)
    log_dict_h_edge = get_logs(log_dirs_h_edge,num_iter=n_iter,out_filename_surfixs=[out_filename_surfixs[0]],step_size=step_size)
            
           
    alphas = list()
    corrs = list()
    mean_scores_list = list()
    vars_list = list()
    for (smooth,mask_correct,normalize) in tqdm(experiment_list):
        beta_alpha = list()
        beta_corr  = list()
        beta_mean_scores = list()
        beta_var = list()
        beta_list = [1,2,3]
        for beta_id in beta_list:
            if beta_id!=1 and not normalize:
                beta_alpha.append(k_alpha)
                beta_mean_scores.append(mean_scores)
                beta_corr.append(corr)
                beta_var.append(k_alpha_var)
                continue
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
            if random == None:
#                 print(scores.shape)
                if baseline=='zeros':
                    scores = np.append(scores,np.zeros([1,scores.shape[-1]]),axis=0)
                elif baseline=='random':
                    scores = np.append(scores,np.random.uniform(scores.min(),scores.max(),[1,scores.shape[-1]]),axis=0)
                inds = np.argsort(scores,axis=0)
#                 print(inds.shape)
                reliability_data = inds.transpose((1,0))
                k_alpha = krippendorff.alpha(reliability_data=reliability_data)
                k_alpha_var = None
            elif random == 'bootstrap':
                num_sample = 10000 if 'ImageNet' not in log_dirs[0] else 5040
                k_alphas = Parallel(n_jobs=N_JOBS,backend='threading')(delayed(get_scores_for_all_logs)(log_dict,log_dirs,n_iter,mask_correct,True,False,None,np.random.choice(num_sample,num_sample),normalize,beta_id,out_filename_surfix) for i in range(10000))
                k_alphas = np.stack(k_alphas,axis=-1)
                k_alpha = k_alphas.mean(-1)
                k_alpha_var = k_alphas.var(-1)
            mean_scores = scores.mean(-1)
            
            corr = stats.spearmanr(scores.T).correlation
#             corr = corr[corr<1]
            beta_alpha.append(k_alpha)
            beta_mean_scores.append(mean_scores)
            beta_corr.append(corr)
            beta_var.append(k_alpha_var)
            # print(corr.shape)
        alphas.append(np.stack(beta_alpha,0))
        mean_scores_list.append(np.stack(beta_mean_scores,0))
        corrs.append(np.stack(beta_corr,0))
        vars_list.append(np.stack(beta_var,0))
    print(alphas)
    
    return alphas,vars_list,mean_scores_list,corrs
def get_scores_for_all_logs(all_log_dirs_dict,log_dirs,num_iter=101,mask_correct=True,use_class=False,use_gaussian=True,class_num=None,subset_inds = None,normalize=True,beta_id=2,out_filename_surfix='',baseline=None):
    scores = np.stack([get_auc_preload(all_log_dirs_dict,l,mask_correct=mask_correct,use_class=True,use_gaussian=False,class_num=None,normalize=normalize,beta_id=beta_id,subset_inds=subset_inds,out_filename_surfix=out_filename_surfix) for l in log_dirs],0)
    start = time.time()
    inds = np.argsort(scores,axis=0)
#     print('sort:',time.time() - start)
#     start = time.time()
    reliability_data = inds.transpose((1,0))
    
    k_alpha = krippendorff.alpha(reliability_data=reliability_data)
#     print('k:',time.time() - start)
#     start = time.time()
    return k_alpha

def get_results_ImageNet(out_file='ImageNet_cmean_morf.pickle',input_dir='./log',random = None):
    # ImageNet
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

    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=int(25*25*0.30773092734737745),out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,random = random)
    mean_result = np.array(results[0])

    if random is not None:
        ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
        for i in range(8):
            to_print = ''
            for j in range(3):
                to_print+=f'{mean_result[i][j]:.2f} ({ci_result[0][i][j]:.2f},{ci_result[1][i][j]:.2f}) & '
            print(to_print)
    with open(out_file,'wb') as f:
        pickle.dump(results,f)

def get_results_MNIST(out_file='MNIST_cmean_morf.pickle',input_dir='./log',random = None):
    # ImageNet
    log_dirs_h_edge = [
        f'{input_dir}/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100',
        f'{input_dir}/SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100',
        f'{input_dir}/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100',
        f'{input_dir}/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100',
        f'{input_dir}/SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100',
        f'{input_dir}/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100',
            ]
    log_dirs_smooth = [
        f'{input_dir}/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100',
        f'{input_dir}/SimpleCNN_E_MNIST_Saliency_no_iter_test_100',
        f'{input_dir}/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100',
        f'{input_dir}/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100',
        f'{input_dir}/SimpleCNN_E_MNIST_GradientShap_no_iter_test_100',
        f'{input_dir}/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100',
            ]

    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=int(25*25*0.30773092734737745),out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,random = random)
    mean_result = np.array(results[0])

    if random is not None:
        ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
        for i in range(8):
            to_print = ''
            for j in range(3):
                to_print+=f'{mean_result[i][j]:.2f} ({ci_result[0][i][j]:.2f},{ci_result[1][i][j]:.2f}) & '
            print(to_print)
    with open(out_file,'wb') as f:
        pickle.dump(results,f)

def get_results_MNIST(out_file='MNIST_cmean_morf.pickle',input_dir='./log',random = None):
    # ImageNet
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

    results = gen_table(log_dirs_h_edge,log_dirs_smooth,n_iter=int(25*25*0.30773092734737745),out_filename_surfixs=['_mcmean_k0_s0','_mcmean_k3_s1'],step_size=1,random = random)
    mean_result = np.array(results[0])

    if random is not None:
        print('random')
        ci_result=(np.array(results[0])-3.291*np.sqrt(results[1]), np.array(results[0])+3.291*np.sqrt(results[1]))
        for i in range(8):
            to_print = ''
            for j in range(3):
                to_print+=f'{mean_result[i][j]:.2f} ({ci_result[0][i][j]:.2f},{ci_result[1][i][j]:.2f}) & '
            print(to_print)
    with open(out_file,'wb') as f:
        pickle.dump(results,f)

if __name__=='__main__':
    DEBUG=False
    N_JOBS=24
    if len(sys.argv)>1:
        if sys.argv[1]=='ImageNet_MoRF':
            get_results_ImageNet(out_file='ImageNet_cmean_MoRF.pickle',input_dir='./log_MoRF',random = 'bootstrap')
        elif sys.argv[1]=='ImageNet_LeRF':
            get_results_ImageNet(out_file='ImageNet_cmean_LeRF.pickle',input_dir='./log_LeRF',random = 'bootstrap')
        elif sys.argv[1]=='MNIST_LeRF':
            get_results_MNIST(out_file='MNIST_cmean_LeRF.pickle',input_dir='./log_LeRF',random = 'bootstrap')
        elif sys.argv[1]=='MNIST_MoRF':
            get_results_MNIST(out_file='MNIST_cmean_MoRF.pickle',input_dir='./log_MoRF',random = 'bootstrap')
        elif sys.argv[1]=='CIFAR_LeRF':
            get_results_CIFAR(out_file='CIFAR_cmean_LeRF.pickle',input_dir='./log_LeRF',random = 'bootstrap')
        elif sys.argv[1]=='CIFAR_MoRF':
            get_results_MNIST(out_file='CIFAR_cmean_MoRF.pickle',input_dir='./log_MoRF',random = 'bootstrap')
    else:
        print('no selection')
