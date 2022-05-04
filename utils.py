import matplotlib.pyplot as plt
import json
import os
import numpy as np

def get_score_bootstrap(log_dir,num_iter,bootstrap_num = 1000):
    score_means = []
    score_vars = []
    for i in range(num_iter):
        cur_log_dir = os.path.join(log_dir,f'{i}')
        cur_gt_dir = os.path.join(cur_log_dir,'out_gt.npy')
        cur_pred_dir = os.path.join(cur_log_dir,'out_pred.npy')
        cur_pred = np.load(cur_pred_dir)
        cur_gt = np.load(cur_gt_dir)
        cur_values = cur_pred[list(range(len(cur_pred))),cur_gt]
        cur_bootstraps = [np.random.choice(cur_values,bootstrap_num).mean() for i in range(10)]
        score_means.append(np.mean(cur_bootstraps))
        score_vars.append(np.var(cur_bootstraps))
    return score_means,score_vars

def get_auc(log_dir,num_iter=11):
    all_score = []
    for i in range(num_iter):
        cur_log_dir = os.path.join(log_dir,f'{i}')
        cur_gt_dir = os.path.join(cur_log_dir,'out_gt.npy')
        cur_pred_dir = os.path.join(cur_log_dir,'out_pred.npy')
        cur_pred = np.load(cur_pred_dir)
        cur_gt = np.load(cur_gt_dir)
        cur_values = cur_pred#[list(range(len(cur_pred))),cur_gt]
        all_score.append(cur_values)
    all_score = np.stack(all_score,0)
    all_score = np.abs((all_score-all_score[0,:])).mean(-1)
    all_score = np.array([all_score[:,np.random.choice(all_score.shape[0],all_score.shape[0])].mean(-1) for i in range(10000)])
    print(all_score.shape)
    # all_score = np.trapz(all_score.mean(-1),axis=0)
    score_var = all_score.var(0)
    all_score = all_score.mean(0)
    return all_score,score_var

def get_score_disjoint(log_dir,num_iter,num_split=10):
    score_means = []
    score_vars = []
    for i in range(num_iter):
        cur_log_dir = os.path.join(log_dir,f'{i}')
        cur_gt_dir = os.path.join(cur_log_dir,'out_gt.npy')
        cur_pred_dir = os.path.join(cur_log_dir,'out_pred.npy')
        cur_pred = np.load(cur_pred_dir)
        cur_gt = np.load(cur_gt_dir)
        cur_values = cur_pred[list(range(len(cur_pred))),cur_gt]
        splits = np.random.choice(cur_values,len(cur_values),replace=False)
        drop_remain = len(cur_values) % num_split
        splits = splits[:len(cur_values)-drop_remain].reshape((-1,num_split)).mean(0)
        score_means.append(np.mean(splits))
        score_vars.append(np.var(splits))
    return score_means,score_vars
    

def generate_uncertainty_plot(experiment_dir_list,out_path,test_result_file='log_test.json',metric='val_acc',max_num_exp=11,show=False):
    all_experiments = dict()
    for i,f in enumerate(experiment_dir_list):
        score_means,score_vars = get_score_disjoint(f,max_num_exp)
        score_sd = np.sqrt(score_vars)
        x = [f'{i*10}%' for i in range(max_num_exp)]
        plt.plot(x,score_means,label = f.split('/')[-1])
        plt.fill_between(x,score_means-score_sd*1.96,score_means+score_sd*1.96, alpha=0.2)
    legend = plt.legend(loc=3,prop={'size': 6})
    plt.savefig(out_path, format='svg')
    if show:
        plt.show()

def generate_uncertainty_auc_plot(experiment_dir_list,out_path,test_result_file='log_test.json',metric='val_acc',max_num_exp=11,show=False):
    all_experiments = dict()
    for i,f in enumerate(experiment_dir_list):
        score_means,score_vars = get_auc(f,max_num_exp)
        # print(score_means.shape)
        score_sd = np.sqrt(score_vars)
        x = [f'{i*10}%' for i in range(max_num_exp)]
        plt.plot(x,score_means,label = f.split('/')[-1])
        plt.fill_between(x,score_means-score_sd*1.96,score_means+score_sd*1.96, alpha=0.2)
    legend = plt.legend(loc=3,prop={'size': 6})
    plt.savefig(out_path, format='svg')
    if show:
        plt.show()

def generate_plot(experiment_dir_list,out_path,test_result_file='log_test.json',metric='val_acc',max_num_exp=11,show=False):
    all_experiments = dict()
    for i in experiment_dir_list:
        all_iterations = list()
        for j in range(max_num_exp):
            
            cur_exp_path = os.path.join(i,f"{j}",test_result_file)
            if os.path.exists(cur_exp_path):
                with open(cur_exp_path, 'r') as f:
                    log = json.load(f)
                all_iterations.append(log[metric])
            else:
                break
        all_experiments[i] = all_iterations
    for name,item in all_experiments.items():
        x = [f'{i*10}%' for i in range(len(item))]
        name = name.split('/')[-1]
        plt.plot(x,item,label = name)
    legend = plt.legend(loc=3,prop={'size': 6})
    plt.savefig(out_path, format='eps')
    if show:
        plt.show()


def get_auc_(log_dir,num_iter=11,mask_correct=True,use_class=False):
    all_score = []
#     loss = torch.nn.CrossEntropyLoss(reduction='none')
    for i in range(num_iter):
        cur_log_dir = os.path.join(log_dir,f'{i}')
        cur_gt_dir = os.path.join(cur_log_dir,'out_gt.npy')
        cur_pred_dir = os.path.join(cur_log_dir,'out_pred.npy')
        cur_pred = np.load(cur_pred_dir).astype(np.float32)
        cur_gt = np.load(cur_gt_dir)
        cur_values = cur_pred#torch.nn.functional.cross_entropy(torch.from_numpy(cur_pred),torch.from_numpy(cur_gt),reduction='none').numpy()
        if use_class:
            cur_values = np.expand_dims(cur_pred[list(range(len(cur_pred))),cur_gt],-1)
            
        print(cur_values.shape)
        all_score.append(cur_values)
        if i==0:
            mask = cur_pred.argmax(-1)==cur_gt
#     print(cur_pred.argmax(-1),cur_gt)
    all_score = np.stack(all_score,0)
    all_score = np.abs((all_score-all_score[0,:])).mean(-1)
    print(all_score.shape)
    all_score = np.trapz(all_score,axis=0)
    if mask_correct:
        all_score = all_score[mask]
    return all_score
def get_krippendorff_alpha()
    

if __name__=='__main__':
    # generate_plot(['/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_hedge_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedGradCam_hedge_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedGradCam_drop_min_hedge_test'],out_path='./dump/resnet18_cifar10_rd_vs_ggcam_hedge.eps')
    # logs = ['/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedGradCam_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedGradCam_drop_min_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Saliency_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Saliency_drop_min_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_test']
    # generate_plot(logs,out_path='./dump/resnet18_cifar10_no_iter.eps')
    # generate_plot(['/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_drop_both_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_drop_both','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random'],out_path='./dump/resnet18_cifar10_hypothesis.eps')
    # logs = ['/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedBackprop_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_GuidedBackprop_drop_both_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_IntegratedGradients_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_IntegratedGradients_drop_both_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_hedge','/home/yimupan/ECCV2022/SaliencyAnalysis/log/ResNet18_E_CIFAR10_Random_drop_both_hedge']
    # generate_plot(logs,out_path='./dump/resnet18_cifar10_base.eps')
    logs = ['/home/yimupan/ECCV2022/SaliencyAnalysis/log/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/vgg16_E_CIFAR10_InputXGradient_no_iter_test','/home/yimupan/ECCV2022/SaliencyAnalysis/log/vgg16_E_CIFAR10_Random_test']
    generate_uncertainty_auc_plot(logs,out_path='./dump/vgg16_cifar10_uncertainty_auc.svg')