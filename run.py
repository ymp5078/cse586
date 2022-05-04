import torch
from tqdm import tqdm
from torchvision import transforms, models
from config import configs as config_lib
import os
import shutil
import json
import numpy as np

from data import datasets
from model import models as model_lib
from model import preprocessing as transform_lib
from timeit import default_timer as timer
import sys


def save_ckp(state, is_best, best_metric, checkpoint_dir, checkpoint_name):
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    f_path = os.path.join(checkpoint_dir,checkpoint_name)
    log_path = os.path.join(checkpoint_dir,'log.json')
    cur_log = {'latest_checkpoint':checkpoint_name}
    cur_log.update(best_metric)
    with open(log_path, 'w') as f:
        json.dump(cur_log, f)
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir,'best_model.pt') 
        torch.save(state, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer,scaler,model_only=True):
    checkpoint = torch.load(checkpoint_fpath)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(checkpoint)
    if model_only:
        return model
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, checkpoint['epoch'], scaler

def try_resume(checkpoint_dir,model,optimizer,scaler,cur_iter,best_metric_name):
    log_path = os.path.join(checkpoint_dir,f'{cur_iter}','log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
        checkpoint_fpath = os.path.join(checkpoint_dir,f'{cur_iter}',log['latest_checkpoint'])
        model, optimizer, epoch, scaler = load_ckp(checkpoint_fpath=checkpoint_fpath,
                                           model=model,
                                           optimizer=optimizer,
                                           scaler=scaler,
                                           model_only=False)
        print(f'Resume from: {checkpoint_fpath}')
        return model, optimizer, scaler, epoch, log[best_metric_name]
    else:
        return model, optimizer, scaler, -1, 0.

def get_map_files(saliency_map_dir,split,num_to_get=None):
    print(num_to_get)
    mdir = os.path.join(saliency_map_dir,split)
    if os.path.exists(mdir):
        map_files = os.listdir(mdir)
        map_files = [mf for mf in map_files if 'base_map' not in mf]
        if num_to_get is not None:
            map_files.sort(key=lambda s: int(s.split("_")[-1].replace('.npy','')))
            assert num_to_get <= len(map_files),f'num_to_get {num_to_get} must be <= len(map_files) {len(map_files)}'
            map_files =  map_files[:num_to_get]
    else:
        map_files = None
    if map_files is not None:
        print('get map files: ',len(map_files))
    return map_files

def test(config):
    # config = config_lib.get_config(config_name)
    # model
    if config.use_torch_hub:
        if config.model_name == 'vgg16':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
            # model = torch.hub.load(config.init_model_ckpt, "cifar10_vgg16_bn",source='local', pretrained=True)
            
        elif config.model_name == 'ResNet50':
            model = models.resnet50(pretrained = True)
    else:
        model = model_lib.get_model(config.model_name)(config.num_classes)
    ckp_path = os.path.join(config.model_dir,f"{config.iter}",f"best_model.pt",)
    if config.no_retrain:
        ckp_path = os.path.join(config.model_dir,"0",f"best_model.pt")
    try:
        model = load_ckp(ckp_path, model, None,None, model_only=True)
    except:
        print(f'Model not loaded from {ckp_path}')
    # print(model)
    if config.use_gpu:
        model.cuda()
    transform = transform_lib.get_preprocessing(config.dataset_name).eval_transforms
    criterion = config.criterion()
    if 'test' in config.saliency_splits:
        map_files = get_map_files(config.saliency_map_dir,'test',config.num_to_get)
        dataset = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,map_dir=config.saliency_map_dir,smooth_map=config.smooth_map,iter_map=config.iter_map,map_files=map_files,split='test',kernel_size=config.kernel_size,sigma=config.sigma,bootstrap_maps=config.bootstrap_maps,use_latest_map=config.use_latest_map,use_basemap=config.use_basemap,cur_iter=config.iter,pertub=config.pertub)
    elif 'val' in config.saliency_splits:
        map_files = get_map_files(config.saliency_map_dir,'val',config.num_to_get)
        dataset = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,map_dir=config.saliency_map_dir,smooth_map=config.smooth_map,iter_map=config.iter_map,map_files=map_files,split='val',kernel_size=config.kernel_size,sigma=config.sigma,bootstrap_maps=config.bootstrap_maps,use_latest_map=config.use_latest_map,use_basemap=config.use_basemap,cur_iter=config.iter,pertub=config.pertub)
    else:
        dataset = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,smooth_map=config.smooth_map,split='test',kernel_size=config.kernel_size,sigma=config.sigma,cur_iter=config.iter,pertub=config.pertub)
    # if os.path.exists(config.saliency_map_dir):
    #     map_files = os.listdir(config.saliency_map_dir)
    # else:
    #     map_files = None
    # map_files =  # need to define num_to_get in the loop
    dataset_maps = [datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,map_dir=config.saliency_map_dir,smooth_map=config.smooth_map,iter_map=config.iter_map,map_files=get_map_files(config.saliency_map_dir,split,config.num_to_get),kernel_size=config.kernel_size,sigma=config.sigma,split=split,cur_iter=config.iter,pertub=config.pertub) for split in config.saliency_splits]
    num_samples = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=5,prefetch_factor=10)
    
    loss_train = 0
    loss_val = 0
    acc_train = 0
    acc_val = 0

    model.train(False)
    model.eval()
    out_data = []
    ground_truth = []
    print("Testing model")
    with tqdm(total=num_samples//config.batch_size) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                    
                inputs, labels = data
                # print([input.mean(0) for input in inputs])
                # print([input.mean(1) for input in inputs])
                ground_truth.append(labels.numpy())
                if config.bootstrap_maps and isinstance(inputs,list):
                    if config.use_gpu:
                        inputs, labels = [torch.autograd.Variable(img.cuda()) for img in inputs], torch.autograd.Variable(labels.cuda())
                    else:
                        inputs, labels = [torch.autograd.Variable(img) for img in inputs], torch.autograd.Variable(labels)
                    with torch.cuda.amp.autocast(enabled=config.use_amp):
                        outputs = [model(img) for img in inputs]
                        
                        preds = torch.stack([torch.max(output.data, 1)[1] for output in outputs])
                        loss = torch.mean(torch.stack([criterion(output, labels) for output in outputs ]))
                    out_data.append(np.concatenate([output.data.cpu().detach().numpy() for output in outputs],axis=0))
                    loss_train += loss.data
                    acc_train += torch.sum(preds == labels.data)//len(inputs)
                else:
                    if config.use_gpu:
                        inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
                    else:
                        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                    with torch.cuda.amp.autocast(enabled=config.use_amp):
                        # print(inputs.type())
                        outputs = model(inputs)
                        
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                    out_data.append(outputs.data.cpu().detach().numpy())
                
                
                    loss_train += loss.data
                    
                    acc_train += torch.sum(preds == labels.data)
                # if i == 0:
                #     for name, param in model.named_parameters():print(name, param,param.dtype)
                #     print(preds,labels.data)
                # print(acc_train,num_samples)
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                pbar.update(1)
    # save the prediction result for finding CI
    ground_truth = np.concatenate(ground_truth,0)
    out_data = np.concatenate(out_data,0)

    out_filename_surfix = f'_m{config.pertub}_k{config.kernel_size}_s{int(config.sigma)}' if config.smooth_map else f'_m{config.pertub}_k{0}_s{0}'
    out_ground_truth_path = os.path.join(config.model_dir,f"{config.iter}",f'out_gt{out_filename_surfix}.npy')
    out_pred_path = os.path.join(config.model_dir,f"{config.iter}",f'out_pred{out_filename_surfix}.npy')
    # out_ground_truth_path = os.path.join(config.model_dir,f"{config.iter}",'out_gt.npy')
    # out_pred_path = os.path.join(config.model_dir,f"{config.iter}",'out_pred.npy')
    np.save(out_ground_truth_path,ground_truth)
    np.save(out_pred_path,out_data)
    # * 2 as we only used half of the dataset
    avg_loss = loss_train / num_samples
    avg_acc = acc_train / num_samples
    log_path = os.path.join(config.model_dir,f"{config.iter}",'log_test.json')
    with open(log_path, 'w') as f:
        json.dump({'val_loss':avg_loss.cpu().detach().numpy().tolist(),'val_acc':avg_acc.cpu().detach().numpy().tolist()}, f)
    print(f'val_loss: {avg_loss}, val_acc: {avg_acc}')
    print('-' * 10)
    if config.gen_map:
        start = timer()
        for dataset_map in dataset_maps:
            if config.use_saliency:
                if not os.path.exists(config.saliency_map_dir):
                    os.mkdir(config.saliency_map_dir)
                mdir = os.path.join(config.saliency_map_dir,dataset_map.split)
                if not os.path.exists(mdir):
                    os.mkdir(mdir)
                saliency_maps = os.listdir(mdir)
                cur_iter = config.iter
                map_path = os.path.join(mdir,f"{config.model_name}_{config.dataset_name}_{cur_iter}")
                dataset_map.generate_discrete_saliency_map(model,map_alg=config.map_alg, map_path=map_path,threshold=config.threshold,drop_max=config.drop_max,use_cuda=config.use_gpu)
                # dataset_map.generate_discrete_saliency_map_parallel(model,map_alg=config.map_alg, map_path=map_path,threshold=config.threshold,drop_max=config.drop_max,use_cuda=config.use_gpu)
            else:
                if not os.path.exists(config.saliency_map_dir):
                    os.mkdir(config.saliency_map_dir)
                mdir = os.path.join(config.saliency_map_dir,dataset_map.split)
                if not os.path.exists(mdir):
                    os.mkdir(mdir)
                saliency_maps = os.listdir(mdir)
                cur_iter = config.iter
                map_path = os.path.join(mdir,f"{config.model_name}_{config.dataset_name}_{cur_iter}")
                dataset_map.generate_discrete_random_saliency_map(map_path=map_path,threshold=config.threshold)
            if cur_iter > 2 and cur_iter not in config.keep_iters:
                dataset_map.remove_maps(os.path.join(mdir,f"{config.model_name}_{config.dataset_name}_{cur_iter-2}"))
        end = timer()
        print(f'{end - start} sec')
def eval_val(model,dataset,config):
    criterion = config.criterion()
    transform = transform_lib.get_preprocessing(config.dataset_name).eval_transforms
    # dataset = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,split='val')
    num_samples = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=5)
    
    loss_train = 0
    loss_val = 0
    acc_train = 0
    acc_val = 0

    model.train(False)
    model.eval()
    print("Evaluating model")
    with torch.no_grad():
        with tqdm(total=num_samples//config.batch_size) as pbar:
            for i, data in enumerate(dataloader):
                    
                inputs, labels = data
                
                if config.use_gpu:
                    inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
                else:
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                
                
                loss_train += loss.data
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                pbar.update(1)
    # * 2 as we only used half of the dataset
    avg_loss = loss_train / num_samples
    avg_acc = acc_train / num_samples
    print(f'val_loss: {avg_loss}, val_acc: {avg_acc}')
    print('-' * 10)
    return {'val_loss':avg_loss.cpu().detach().numpy().tolist(),'val_acc':avg_acc.cpu().detach().numpy().tolist()}

def train(config):
    # config = config_lib.get_config(config_name)
    # model
    model = model_lib.get_model(config.model_name)(config.num_classes)
    if config.keep_weights and config.iter!=9:
        
        ckp_path = os.path.join(config.model_dir,f"{config.iter}",f"best_model.pt")
        if not os.path.exists(ckp_path):
            print('load weights from previous iter')
            ckp_path = os.path.join(config.model_dir,f"{config.iter+1}",f"best_model.pt")
            model = load_ckp(ckp_path, model, None,model_only=True)
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if config.use_gpu:
        model.cuda()
    
    
    # training
    criterion = config.criterion()
    optimizer = config.optimizer(model.parameters(),lr=config.lr,momentum=config.momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    best_metric = 0.
    model, optimizer, scaler, cur_epoch, best_metric = try_resume(checkpoint_dir=config.model_dir,model=model,optimizer=optimizer,scaler=scaler,cur_iter=config.iter,best_metric_name=config.metric)
    lr_scheduler = config.lr_scheduler(optimizer, step_size=config.step_size,gamma=config.gamma)

    # TODO: put this into the dataset class
    transform = transform_lib.get_preprocessing(config.dataset_name).train_transforms
    # if os.path.exists(config.saliency_map_dir):
    #     map_files = os.listdir(config.saliency_map_dir)
    # else:
    #     map_files = None
    # data
    map_files = get_map_files(config.saliency_map_dir,'train',config.num_to_get)
    dataset_train = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,map_files=map_files,map_dir=config.saliency_map_dir,smooth_map=config.smooth_map,split='train',map_out=config.map_out,kernel_size=config.kernel_size,sigma=config.sigma)
    dataset_val = datasets.get_dataset(config.dataset_name)(config.dataset_dir,transform=transform,split='val',kernel_size=config.kernel_size,sigma=config.sigma)
    num_samples = len(dataset_train)
    dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=config.batch_size,shuffle=True,num_workers=5)
    
    for epoch in range(cur_epoch+1,config.num_epoches):
        print("Epoch {}/{}".format(epoch, config.num_epoches))
        print('-' * 10)
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        model.train(True)
        with tqdm(total=num_samples//config.batch_size) as pbar:
            for i, data in enumerate(dataloader):
                    
                inputs, labels = data
                
                if config.use_gpu:
                    inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
                else:
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                
                
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    outputs = model(inputs)
                
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # optimizer.step()
                
                loss_train += loss.data
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                pbar.update(1)
        # print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / num_samples
        avg_acc = acc_train / num_samples
        print(f'loss: {avg_loss}, acc: {avg_acc}')
        print()
        model.train(False)
        model.eval()
        
        is_best = False
        cur_metric = {'loss':avg_loss.cpu().detach().numpy().tolist(),'acc':avg_acc.cpu().detach().numpy().tolist()}
        if epoch%config.eval_freq==0:
            cur_metric.update(eval_val(model,dataset_val,config))
            if cur_metric[config.metric] > best_metric:
                best_metric = cur_metric[config.metric]
                is_best=True
                print(f'save best at epoch {epoch}')
        log_file = f"{config.model_name}_{config.dataset_name}.pt"
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }
        checkpoint_dir = os.path.join(config.model_dir,f"{config.iter}")
        
        save_ckp(checkpoint,is_best=is_best,best_metric=cur_metric, checkpoint_dir=checkpoint_dir, checkpoint_name=log_file)
        # else:
        #     log_file = f"{config.model_name}_{config.dataset_name}.pt"
        #     checkpoint = {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     }
        #     checkpoint_dir = os.path.join(config.model_dir,f"{config.iter}")
        #     save_ckp(checkpoint,is_best=False,best_metric=best_metric, checkpoint_dir=checkpoint_dir, checkpoint_name=log_file)



def train_multi_iters(config_name):
    config = config_lib.get_config(config_name)
    config.num_to_get = None
    iterration_log_path = os.path.join(config.model_dir,'iter_log.json')
    if os.path.exists(iterration_log_path):
        with open(iterration_log_path, 'r') as f:
            log = json.load(f)
        cur_iter = log['iter']
        print(f'found iter_log.json, resume iteration {cur_iter}')
    else:
        cur_iter = 0
        os.mkdir(config.model_dir)
    for i in range(cur_iter,config.num_iters):
        config.iter = i
        with open(iterration_log_path, 'w') as f:
            json.dump({'iter':i}, f)
        iter_path = os.path.join(config.model_dir,f'{i}')
        if not os.path.exists(iter_path):
            os.mkdir(iter_path)
        train(config)
        test(config)

def train_on_map_multi_iters(num_iters,config_name):
    config = config_lib.get_config(config_name)
    iterration_log_path = os.path.join(config.model_dir,'iter_log.json')
    if os.path.exists(iterration_log_path):
        with open(iterration_log_path, 'r') as f:
            log = json.load(f)
        cur_iter = log['iter']
        print(f'found iter_log.json, resume iteration {cur_iter}')
    else:
        cur_iter = 0
        os.mkdir(config.model_dir)
    for i in range(cur_iter,config.num_iters):
        config.iter = i
        config.num_to_get = i+1
        with open(iterration_log_path, 'w') as f:
            json.dump({'iter':i}, f)
        iter_path = os.path.join(config.model_dir,f'{i}')
        if not os.path.exists(iter_path):
            os.mkdir(iter_path)
        train(config)
        test(config)

def train_progressive_dropping(config_name):
    config = config_lib.get_config(config_name)
    
    iterration_log_path = os.path.join(config.model_dir,'iter_log.json')
    if os.path.exists(iterration_log_path):
        with open(iterration_log_path, 'r') as f:
            log = json.load(f)
        cur_iter = log['iter']
        print(f'found iter_log.json, resume iteration {cur_iter}')
    else:
        cur_iter = 0
        os.mkdir(config.model_dir)
    # for i in range(cur_iter,num_iters):
    for i in range(config.num_iters-1,cur_iter-1,-1):
        config.iter = i
        config.num_to_get = i
        if i > 0:
            config.num_epoches=10
        with open(iterration_log_path, 'w') as f:
            json.dump({'iter':i}, f)
        iter_path = os.path.join(config.model_dir,f'{i}')
        if not os.path.exists(iter_path):
            os.mkdir(iter_path)
        train(config)
        test(config)

def test_multi_iters(config_name,init_model_ckpt,pertubs=None,overwrite=None):
    config = config_lib.get_config(config_name)
    config.num_to_get = None
    config.keep_iters = [99,config.num_iters-1]
    out_filename_surfix = f'_m{config.pertub}_k{config.kernel_size}_s{int(config.sigma)}' if config.smooth_map else f'_m{config.pertub}_k{0}_s{0}'
    if overwrite is not None:
        for (k,v) in overwrite.items():
            setattr(config, k, v)
    if overwrite['drop_max']:
        config.model_dir = config.model_dir.replace('LeRF','MoRF')
        config.saliency_map_dir = config.saliency_map_dir.replace('LeRF','MoRF')
    else:
        config.model_dir = config.model_dir.replace('MoRF','LeRF')
        config.saliency_map_dir = config.saliency_map_dir.replace('MoRF','LeRF')
    
    if pertubs is not None:
        out_filename_surfixs = [f'_m{pertub}_k{config.kernel_size}_s{int(config.sigma)}' if config.smooth_map else f'_m{pertub}_k{0}_s{0}' for pertub in pertubs]
        iterration_log_paths = [os.path.join(config.model_dir,f'iter_log_{surfix}.json') for surfix in out_filename_surfixs]
    else:
        iterration_log_paths = [os.path.join(config.model_dir,f'iter_log_{out_filename_surfix}.json')]
        pertubs = [config.pertub]
    for (p_idx, pertub) in enumerate(pertubs):
        config.pertub = pertub
        iterration_log_path = iterration_log_paths[p_idx]
        if os.path.exists(iterration_log_path):
            with open(iterration_log_path, 'r') as f:
                log = json.load(f)
            cur_iter = log['iter']
            print(f'found iter_log.json, resume iteration {cur_iter}')
        else:
            if 'test' in config.saliency_splits:

                map_files = get_map_files(config.saliency_map_dir,'test',config.num_to_get)
            elif 'val' in config.saliency_splits:

                map_files = get_map_files(config.saliency_map_dir,'val',config.num_to_get)
            
            cur_iter = 0 if map_files is None or len(map_files)==0 else 1
            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)
        init_path = os.path.join(config.model_dir,'0')
        print(init_path)
        if not os.path.exists(init_path) and init_model_ckpt is not None:
            os.mkdir(init_path)
            print(f'Setting up the starting model from {init_model_ckpt}')
            init_model_path = os.path.join(init_path,'best_model.pt')
            shutil.copyfile(init_model_ckpt,init_model_path)
        config.init_model_ckpt = init_model_ckpt
    
        for i in range(cur_iter,config.num_iters):
            config.iter = i
            with open(iterration_log_path, 'w') as f:
                json.dump({'iter':i}, f)
            iter_path = os.path.join(config.model_dir,f'{i}')
            if not os.path.exists(iter_path):
                os.mkdir(iter_path)
            print(f'perform {pertub} at iter {i}')
            test(config)

if __name__=='__main__':
    # train_multi_iters('SimpleCNN_E_MNIST_Random')
    overwrite = {'drop_max':False}# LeRF
    print(sys.argv)
    if len(sys.argv)>1:
        
        
        if 'MNIST' in sys.argv[1]:
            print('MNIST')
            overwrite = {'drop_max':False}
            test_multi_iters(sys.argv[1],'log_LeRF2/SimpleCNN_E_MNIST_Random/0/SimpleCNN_E_MNIST.pt',['cmean'],overwrite=overwrite)
            # test_multi_iters(sys.argv[1],None,['cmean'],overwrite=overwrite)#,'noise'])
            # overwrite = {'drop_max':True}
            # test_multi_iters(sys.argv[1],'log_LeRF2/SimpleCNN_E_MNIST_Random/0/SimpleCNN_E_MNIST.pt',['cmean'],overwrite=overwrite)#,'noise'])
        else:
            overwrite = {'drop_max':False}
            test_multi_iters(sys.argv[1],None,['cmean'],overwrite=overwrite)#,'noise'])
            overwrite = {'drop_max':True}
            test_multi_iters(sys.argv[1],None,['cmean'],overwrite=overwrite)#,'noise'])
    else:
        # 100 iterations
        # test_multi_iters('vgg16_E_CIFAR10_InputXGradient_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])#
        # test_multi_iters('vgg16_E_CIFAR10_Saliency_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])

        # # # add more
        # test_multi_iters('vgg16_E_CIFAR10_GradientShap_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_Deconvolution_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])

        # test_multi_iters('vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/weights/cifar10_vgg16_bn-6ee7ea24.pt',['cmean'],overwrite=overwrite)#,'noise'])

        # test_multi_iters('SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_GradientShap_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        
        # test_multi_iters('SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        # test_multi_iters('SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)
        

        # test_multi_iters('SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('SimpleCNN_E_MNIST_Saliency_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100','/home/yimupan/ECCV2022/SaliencyAnalysis/log/SimpleCNN_E_MNIST_Random/0/best_model.pt',['cmean'],overwrite=overwrite)#,'noise'])
        
        # test_multi_iters('ResNet50_E_ImageNet_InputXGradient_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('ResNet50_E_ImageNet_Saliency_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('ResNet50_E_ImageNet_IntegratedGradients_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])

        # test_multi_iters('ResNet50_E_ImageNet_GradientShap_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('ResNet50_E_ImageNet_GuidedBackprop_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        # test_multi_iters('ResNet50_E_ImageNet_Deconvolution_val_100',None,['cmean'],overwrite=overwrite)#,'noise'])


        # test_multi_iters('ResNet50_E_ImageNet_InputXGradient_val_h_edge_100',None,['cmean'],overwrite=overwrite)
        # test_multi_iters('ResNet50_E_ImageNet_Saliency_val_h_edge_100',None,['cmean'],overwrite=overwrite)
        # test_multi_iters('ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100',None,['cmean'],overwrite=overwrite)

        test_multi_iters('ResNet50_E_ImageNet_GradientShap_val_h_edge_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        test_multi_iters('ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100',None,['cmean'],overwrite=overwrite)#,'noise'])
        test_multi_iters('ResNet50_E_ImageNet_Deconvolution_val_h_edge_100',None,['cmean'],overwrite=overwrite)#,'noise'])
