import torch
from captum.attr import (
    GuidedGradCam,#
    GradientShap,
    KernelShap,
    InputXGradient,#
    Saliency,#
    IntegratedGradients,#
    GuidedBackprop,
    DeepLiftShap,#
    DeepLift,
    LRP, # error 
    Occlusion,
    Lime,
    ShapleyValues, # not use too slow
    FeatureAblation, # not use no reference
    ShapleyValueSampling, # not use too slow
    Deconvolution)


class SimpleCNN:
    """
        This class is for inherit only
    """
    model_name = 'SimpleCNN'
    criterion = torch.nn.CrossEntropyLoss
    batch_size = 256
    num_epoches = 30
    num_classes = 10
    eval_freq = 1
    use_gpu = True
    use_amp = False
    use_torch_hub = False
    # optimizer
    lr = 0.1
    momentum=0.9
    optimizer = torch.optim.SGD
    
    # lr _scheduler
    step_size=7
    gamma=0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR

    # metric
    metric = 'val_acc'
    
class SimpleCNN_E_MNIST(SimpleCNN):
    dataset_name='E_MNIST'
    dataset_dir = './data/MNIST'
    gen_map = True
    keep_weights = False
    num_iters = 12
    map_out = True
    smooth_map = True
    drop_max = True
    kernel_size=3
    sigma=1.0
    iter_map = True
    bootstrap_maps = False
    use_latest_map = False
    use_basemap = False
    pertub = 'cmean'
    # pertub = 'noise'
    num_iters = 212
    # drop related

class SimpleCNN_E_MNIST_Random(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Random'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/MNIST/SimpleCNN_Random_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = False
    saliency_splits = ['train']
    # cam
    # cam = GradCAM

class SimpleCNN_E_MNIST_Random_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Random_test_100'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/MNIST/SimpleCNN_Random_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    # cam = GradCAM
################
class SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_ShapleyValueSampling_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = ShapleyValueSampling
    iter_map = False

class SimpleCNN_E_MNIST_GradientShap_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GradientShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GradientShap_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GradientShap
    iter_map = False


class SimpleCNN_E_MNIST_KernelShap_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_KernelShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_KernelShap_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = KernelShap
    iter_map = False

class SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GuidedBackprop_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedBackprop
    iter_map = False

class SimpleCNN_E_MNIST_DeepLift_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_DeepLift_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_DeepLift_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLift
    iter_map = False

class SimpleCNN_E_MNIST_LRP_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_LRP_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_LRP_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = LRP
    iter_map = False

class SimpleCNN_E_MNIST_Occlusion_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Occlusion_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Occlusion_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Occlusion
    iter_map = False

class SimpleCNN_E_MNIST_Lime_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Lime_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Lime_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Lime
    iter_map = False

class SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_ShapleyValues_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = ShapleyValues
    iter_map = False

class SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_FeatureAblation_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = FeatureAblation
    iter_map = False

class SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Deconvolution_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Deconvolution
    iter_map = False

###########

################
class SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_ShapleyValueSampling_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = ShapleyValueSampling
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GradientShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GradientShap
    iter_map = False
    smooth_map = False


class SimpleCNN_E_MNIST_KernelShap_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_KernelShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_KernelShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = KernelShap
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GuidedBackprop_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedBackprop
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_DeepLift_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_DeepLift_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_DeepLift_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLift
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_LRP_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_LRP_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_LRP_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = LRP
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_Occlusion_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Occlusion_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Occlusion_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Occlusion
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_Lime_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Lime_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Lime_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Lime
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_ShapleyValues_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = ShapleyValues
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_FeatureAblation_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = FeatureAblation
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Deconvolution_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Deconvolution
    iter_map = False
    smooth_map = False

###########

class SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GuidedGradCam_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False

class SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_InputXGradient_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False

class SimpleCNN_E_MNIST_Saliency_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Saliency_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Saliency_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False

class SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_IntegratedGradients_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False

class SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_DeepLiftShap_no_iter_test_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False


class SimpleCNN_E_MNIST_Random_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Random_test_h_edge_100'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/MNIST/SimpleCNN_Random_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    smooth_map = False
    # cam
    # cam = GradCAM

class SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_GuidedGradCam_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_InputXGradient_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_Saliency_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_IntegratedGradients_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    smooth_map = False

class SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_h_edge_100(SimpleCNN_E_MNIST):
    model_dir = './log_LeRF_ImageNet/SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/MNIST/SimpleCNN_DeepLiftShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    # num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    smooth_map = False

class vgg16:
    """
        This class is for inherit only
    """
    model_name = 'vgg16'
    criterion = torch.nn.CrossEntropyLoss
    batch_size = 256
    num_epoches = 100
    num_classes = 10
    eval_freq = 1
    use_gpu = True
    use_amp = False
    use_torch_hub = True
    # optimizer
    lr = 0.001
    momentum=0.9
    optimizer = torch.optim.SGD
    
    # lr _scheduler
    step_size=7
    gamma=0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR

    # metric
    metric = 'val_acc'



class vgg16_E_CIFAR10(vgg16):
    dataset_name='E_CIFAR10'
    dataset_dir = './data/CIFAR10'
    gen_map = True
    keep_weights = False
    num_iters = 12
    map_out = True
    smooth_map = True
    drop_max = True
    kernel_size=3
    sigma=1.0
    iter_map = True
    bootstrap_maps = False
    use_latest_map = False
    use_basemap = False
    pertub = 'cmean'
    # pertub = 'noise'
    # drop related

class vgg16_E_CIFAR10_GradCAM(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradCAM'
    # saliency map
    use_saliency = True
    smooth_map = True
    saliency_map_dir = './data/CIFAR10/vgg16_GradCAM_saliency_maps_LeRF'
    threshold = 0.1
    # cam
    cam = GuidedGradCam
    map_out = True
    
    
class vgg16_E_CIFAR10_GradCAM_squeeze(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradCAM_squeeze'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GradCAM_squeeze_saliency_maps_LeRF'
    threshold = 0.1
    # cam
    cam = GuidedGradCam
    map_out = True

class vgg16_E_CIFAR10_GradCAM_on_map(vgg16_E_CIFAR10_GradCAM):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradCAM_on_map'
    map_out = False # set this to False to keep the high saliency area, otherwise mask out the high saliency area
    gen_map = False

class vgg16_E_CIFAR10_GradCAM_progressive(vgg16_E_CIFAR10_GradCAM):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradCAM_progressive'
    map_out = True # set this to False to keep the high saliency area, otherwise mask out the high saliency area
    gen_map = False
    keep_weights = True

class vgg16_E_CIFAR10_Random(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = False
    saliency_split = 'train'
    # cam
    # cam = GradCAM

class vgg16_E_CIFAR10_Random_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    # cam = GradCAM



class vgg16_E_CIFAR10_Random_test_boot(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test_boot'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_boot_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    bootstrap_maps = True

class vgg16_E_CIFAR10_Random_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test_use_latest_map'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    use_latest_map = True


class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_saliency_maps_LeRF'
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False

class vgg16_E_CIFAR10_GuidedGradCam_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = True


class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_boot(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_boot'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_boot_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    bootstrap_maps = True

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_basemap(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_basemap'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_basemap_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    num_iters = 2
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    bootstrap_maps = False
    use_basemap = True

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_use_latest_map'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    use_latest_map = True

class vgg16_E_CIFAR10_InputXGradient_no_iter_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False

class vgg16_E_CIFAR10_InputXGradient_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = True

class vgg16_E_CIFAR10_InputXGradient_no_iter_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test_use_latest_map'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False
    use_latest_map = True

class vgg16_E_CIFAR10_InputXGradient_no_iter_test_basemap(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test_basemap'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_basemap_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    num_iters = 2
    # cam
    map_alg = InputXGradient
    iter_map = False
    bootstrap_maps = False
    use_basemap = True

class vgg16_E_CIFAR10_Saliency_no_iter_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False

class vgg16_E_CIFAR10_Saliency_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = True

class vgg16_E_CIFAR10_Saliency_no_iter_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test_use_latest_map'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False
    use_latest_map = True

class vgg16_E_CIFAR10_Saliency_no_iter_test_basemap(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test_basemap'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_basemap_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    num_iters = 2
    # cam
    map_alg = Saliency
    iter_map = False
    bootstrap_maps = False
    use_basemap = True

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False

class vgg16_E_CIFAR10_IntegratedGradients_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = True

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_use_latest_map'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    use_latest_map = True

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_basemap(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_basemap'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_basemap_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    num_iters = 2
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    bootstrap_maps = False
    use_basemap = True

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False

class vgg16_E_CIFAR10_DeepLiftShap_test(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_test'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_test_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = True
    

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_use_latest_map(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_use_latest_map'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_use_latest_map_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    use_latest_map = True

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_basemap(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_basemap'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_basemap_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    num_iters = 2
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    bootstrap_maps = False
    use_basemap = True

# val
class vgg16_E_CIFAR10_Random_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_val'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    # cam = GradCAM

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_val'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GuidedGradCam
    iter_map = False

class vgg16_E_CIFAR10_InputXGradient_no_iter_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_val'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = InputXGradient
    iter_map = False

class vgg16_E_CIFAR10_Saliency_no_iter_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_val'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Saliency
    iter_map = False

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_val'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = IntegratedGradients
    iter_map = False

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_val(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_val'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_val_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = DeepLiftShap
    iter_map = False


# 100 iterations 

class vgg16_E_CIFAR10_Random_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test_100'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    # cam = GradCAM

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False

class vgg16_E_CIFAR10_InputXGradient_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False

class vgg16_E_CIFAR10_Saliency_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False

# iter
class vgg16_E_CIFAR10_GuidedGradCam_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = True

class vgg16_E_CIFAR10_InputXGradient_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = True

class vgg16_E_CIFAR10_Saliency_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = True

class vgg16_E_CIFAR10_IntegratedGradients_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = True

class vgg16_E_CIFAR10_DeepLiftShap_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = True

#####################
class vgg16_E_CIFAR10_GradientShap_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradientShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GradientShap_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GradientShap
    iter_map = False

class vgg16_E_CIFAR10_KernelShap_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_KernelShap_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_KernelShap_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = KernelShap
    iter_map = False

class vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedBackprop_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedBackprop
    iter_map = False

class vgg16_E_CIFAR10_DeepLift_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLift_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLift_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLift
    iter_map = False

class vgg16_E_CIFAR10_LRP_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_LRP_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_LRP_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = LRP
    iter_map = False

class vgg16_E_CIFAR10_Occlusion_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Occlusion_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Occlusion_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Occlusion
    iter_map = False

class vgg16_E_CIFAR10_Lime_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Lime_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Lime_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Lime
    iter_map = False

class vgg16_E_CIFAR10_Deconvolution_no_iter_test_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Deconvolution_no_iter_test_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Deconvolution_no_iter_test_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001   
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Deconvolution
    iter_map = False

#####################
    

class vgg16_E_CIFAR10_Random_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test_h_edge'
    # saliency map
    use_saliency = False
    smooth_map = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_h_edge_saliency_maps_LeRF'
    threshold = 0.1 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    smooth_map = False


# 100 iteration

class vgg16_E_CIFAR10_Random_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Random_test_h_edge_100'
    # saliency map
    use_saliency = False
    smooth_map = False
    saliency_map_dir = './data/CIFAR10/vgg16_Random_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']

class vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedGradCam_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_InputXGradient_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = InputXGradient
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Saliency_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Saliency
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_IntegratedGradients_no_iter_test_100_h_edge_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001  
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLiftShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    smooth_map = False

#####################
class vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GradientShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GradientShap
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_KernelShap_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_KernelShap_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_KernelShap_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = KernelShap
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_GuidedBackprop_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = GuidedBackprop
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_DeepLift_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_DeepLift_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_DeepLift_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = DeepLift
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_LRP_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_LRP_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_LRP_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = LRP
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_Occlusion_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Occlusion_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Occlusion_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Occlusion
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_Lime_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Lime_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Lime_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Lime
    iter_map = False
    smooth_map = False

class vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100(vgg16_E_CIFAR10):
    model_dir = './log_LeRF_ImageNet/vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/CIFAR10/vgg16_Deconvolution_no_iter_test_h_edge_100_saliency_maps_LeRF'
    num_iters = 455
    threshold = 0.001 
    no_retrain = True
    saliency_splits = ['test']
    # cam
    map_alg = Deconvolution
    iter_map = False
    smooth_map = False
#####################


class vgg16_E_Food101(vgg16):
    batch_size=10
    num_classes = 101
    dataset_name='E_Food101'
    dataset_dir = './data/Food101'
    gen_map = True
    map_out = True
    keep_weights = False
    num_iters = 10
    kernel_size=5
    sigma=5.0

class vgg16_E_Food101_Random(vgg16_E_Food101):
    model_dir = f'./log_LeRF_ImageNet/{vgg16_E_Food101.model_name}_{vgg16_E_Food101.dataset_name}_Random'
    # saliency map
    use_saliency = False
    saliency_map_dir = f'{vgg16_E_Food101.dataset_dir}/vgg16_Random_saliency_maps_LeRF'
    threshold = 0.1 
    map_out = True

class ResNet18:
    """
        This class is for inherit only
    """
    model_name = 'ResNet18'
    criterion = torch.nn.CrossEntropyLoss
    batch_size = 256
    num_epoches = 100
    num_classes = 10
    eval_freq = 1
    use_gpu = True
    use_amp = True
    
    # optimizer
    lr = 0.001
    momentum=0.9
    optimizer = torch.optim.SGD
    
    # lr _scheduler
    step_size=7
    gamma=0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR

    # metric
    metric = 'val_acc'

class ResNet18_E_CIFAR10(ResNet18):
    dataset_name='E_CIFAR10'
    dataset_dir = './data/CIFAR10'
    gen_map = True
    keep_weights = False
    num_iters = 12
    map_out = True
    smooth_map = True
    drop_max = True
    kernel_size=5
    sigma=5.0
    iter_map = True
    # drop related

class ResNet50:
    """
        This class is for inherit only
    """
    model_name = 'ResNet50'
    criterion = torch.nn.CrossEntropyLoss
    batch_size = 128
    num_epoches = 100
    num_classes = 1000
    eval_freq = 1
    use_gpu = True
    use_amp = False
    use_torch_hub = True


    # optimizer
    lr = 0.01
    momentum=0.9
    optimizer = torch.optim.SGD
    
    # lr _scheduler
    step_size=7
    gamma=0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR

    # metric
    metric = 'val_acc'

    saliency_splits = ['train']
    


class ResNet50_E_Food101(ResNet50):
    batch_size=256
    num_classes = 1000
    dataset_name='E_Food101'
    dataset_dir = './data/Food101'
    gen_map = True
    keep_weights = False
    num_iters = 12
    map_out = True
    smooth_map = True
    drop_max = True
    kernel_size=5
    sigma=5.0
    iter_map = True
    bootstrap_maps = False
    use_latest_map = False
    use_basemap = False

class ResNet50_E_ImageNet(ResNet50):
    batch_size=64
    num_classes = 1000
    dataset_name='E_ImageNet'
    dataset_dir = './data/ImageNet'
    gen_map = True
    keep_weights = False
    num_iters = 193
    keep_iters = [100]
    map_out = True
    smooth_map = True
    drop_max = True
    kernel_size=3
    sigma=1.0
    pertub='cmean'
    # pertub = 'noise'
    iter_map = True
    bootstrap_maps = False
    use_latest_map = False
    use_basemap = False

class ResNet50_E_ImageNet_Random_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Random_val_100'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/ImageNet/ResNet50_Random_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    # cam = GradCAM

class ResNet50_E_ImageNet_GuidedGradCam_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GuidedGradCam_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GuidedGradCam_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GuidedGradCam
    iter_map = False

class ResNet50_E_ImageNet_DeepLiftShap_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_DeepLiftShap_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_DeepLiftShap_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = DeepLiftShap
    iter_map = False

class ResNet50_E_ImageNet_InputXGradient_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_InputXGradient_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_InputXGradient_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = InputXGradient
    iter_map = False

class ResNet50_E_ImageNet_Saliency_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Saliency_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Saliency_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Saliency
    iter_map = False

class ResNet50_E_ImageNet_IntegratedGradients_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_IntegratedGradients_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_IntegratedGradients_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = IntegratedGradients
    iter_map = False

#################
class ResNet50_E_ImageNet_GradientShap_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GradientShap_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GradientShap_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GradientShap
    iter_map = False

class ResNet50_E_ImageNet_KernelShap_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_KernelShap_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_KernelShap_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = KernelShap
    iter_map = False

class ResNet50_E_ImageNet_GuidedBackprop_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GuidedBackprop_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GuidedBackprop_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GuidedBackprop
    iter_map = False

class ResNet50_E_ImageNet_DeepLift_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_DeepLift_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_DeepLift_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = DeepLift
    iter_map = False

class ResNet50_E_ImageNet_LRP_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_LRP_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_LRP_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = LRP
    iter_map = False

class ResNet50_E_ImageNet_Lime_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Lime_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Lime_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Lime
    iter_map = False

class ResNet50_E_ImageNet_Deconvolution_val_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Deconvolution_val_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Deconvolution_val_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Deconvolution
    iter_map = False


#################

# hedge

class ResNet50_E_ImageNet_Random_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Random_val_h_edge_100'
    # saliency map
    use_saliency = False
    saliency_map_dir = './data/ImageNet/ResNet50_Random_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    smooth_map = False
    # cam
    # cam = GradCAM

class ResNet50_E_ImageNet_GuidedGradCam_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GuidedGradCam_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GuidedGradCam_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GuidedGradCam
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_DeepLiftShap_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_DeepLiftShap_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_DeepLiftShap_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = DeepLiftShap
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_InputXGradient_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_InputXGradient_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_InputXGradient_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = InputXGradient
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_Saliency_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Saliency_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Saliency_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Saliency
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_IntegratedGradients_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = IntegratedGradients
    iter_map = False
    smooth_map = False

##############
class ResNet50_E_ImageNet_GradientShap_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GradientShap_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GradientShap_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GradientShap
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_KernelShap_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_KernelShap_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_KernelShap_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = KernelShap
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_GuidedBackprop_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = GuidedBackprop
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_DeepLift_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_DeepLift_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_DeepLift_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = DeepLift
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_LRP_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_LRP_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_LRP_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = LRP
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_Lime_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Lime_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Lime_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Lime
    iter_map = False
    smooth_map = False

class ResNet50_E_ImageNet_Deconvolution_val_h_edge_100(ResNet50_E_ImageNet):
    model_dir = './log_LeRF_ImageNet/ResNet50_E_ImageNet_Deconvolution_val_h_edge_100'
    # saliency map
    use_saliency = True
    saliency_map_dir = './data/ImageNet/ResNet50_Deconvolution_val_h_edge_100_saliency_maps_LeRF'
    num_iters = 193
    keep_iters = [100]
    threshold = 0.002  
    no_retrain = True
    saliency_splits = ['val']
    # cam
    map_alg = Deconvolution
    iter_map = False
    smooth_map = False

##############


class ResNet50_E_Food101_Random(ResNet50_E_Food101):
    model_dir = f'./log_LeRF_ImageNet/{ResNet50_E_Food101.model_name}_{ResNet50_E_Food101.dataset_name}_Random'
    # saliency map
    use_saliency = False
    smooth_map = True
    saliency_map_dir = f'{ResNet50_E_Food101.dataset_dir}/ResNet50_Random_saliency_maps_LeRF'
    threshold = 0.1 
    map_out = True

class ResNet50_E_Food101_GradCAM(ResNet50_E_Food101):
    model_dir = f'./log_LeRF_ImageNet/{ResNet50_E_Food101.model_name}_{ResNet50_E_Food101.dataset_name}_GradCAM'
    # saliency map
    use_saliency = True
    smooth_map = True
    saliency_map_dir = f'{ResNet50_E_Food101.dataset_dir}/ResNet50_GradCAM_saliency_maps_LeRF'
    threshold = 0.1 
    map_out = True
    # cam
    cam = GuidedGradCam

class ResNet50_E_Food101_GradCAM_test(ResNet50_E_Food101):
    model_dir = f'./log_LeRF_ImageNet/{ResNet50_E_Food101.model_name}_{ResNet50_E_Food101.dataset_name}_GradCAM_test'
    # saliency map
    use_saliency = True
    smooth_map = True
    saliency_map_dir = f'{ResNet50_E_Food101.dataset_dir}/ResNet50_GradCAM_test_saliency_maps_LeRF'
    threshold = 0.1 
    map_out = True
    no_retrain = True
    # cam
    cam = GuidedGradCam
    saliency_splits = ['test']

class ResNet50_E_Food101_Random_test(ResNet50_E_Food101):
    model_dir = f'./log_LeRF_ImageNet/{ResNet50_E_Food101.model_name}_{ResNet50_E_Food101.dataset_name}_Random_test'
    # saliency map
    use_saliency = False
    smooth_map = True
    saliency_map_dir = f'{ResNet50_E_Food101.dataset_dir}/ResNet50_Random_test_saliency_maps_LeRF'
    threshold = 0.1 
    map_out = True

    no_retrain = True
    saliency_splits = ['test']

CONFIGS = {
    'SimpleCNN_E_MNIST_Random':SimpleCNN_E_MNIST_Random,
    'SimpleCNN_E_MNIST_Random_test_100':SimpleCNN_E_MNIST_Random_test_100,
    'SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100':SimpleCNN_E_MNIST_InputXGradient_no_iter_test_100,
    'SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_100':SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_100,
    'SimpleCNN_E_MNIST_Saliency_no_iter_test_100':SimpleCNN_E_MNIST_Saliency_no_iter_test_100,
    'SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100':SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_100,
    'SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_100':SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_100,
    
    'SimpleCNN_E_MNIST_GradientShap_no_iter_test_100':SimpleCNN_E_MNIST_GradientShap_no_iter_test_100,
    'SimpleCNN_E_MNIST_KernelShap_no_iter_test_100':SimpleCNN_E_MNIST_KernelShap_no_iter_test_100,
    'SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100':SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_100,
    'SimpleCNN_E_MNIST_DeepLift_no_iter_test_100':SimpleCNN_E_MNIST_DeepLift_no_iter_test_100,
    'SimpleCNN_E_MNIST_LRP_no_iter_test_100':SimpleCNN_E_MNIST_LRP_no_iter_test_100,
    'SimpleCNN_E_MNIST_Occlusion_no_iter_test_100':SimpleCNN_E_MNIST_Occlusion_no_iter_test_100,
    'SimpleCNN_E_MNIST_Lime_no_iter_test_100':SimpleCNN_E_MNIST_Lime_no_iter_test_100,
    'SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_100':SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_100,
    'SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_100':SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_100,
    'SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100':SimpleCNN_E_MNIST_Deconvolution_no_iter_test_100,
    'SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_100':SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_100,

    'SimpleCNN_E_MNIST_Random_test_h_edge_100':SimpleCNN_E_MNIST_Random_test_h_edge_100,
    'SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_InputXGradient_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_GuidedGradCam_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_Saliency_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_IntegratedGradients_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_DeepLiftShap_no_iter_test_h_edge_100,

    'SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_GradientShap_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_KernelShap_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_KernelShap_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_GuidedBackprop_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_DeepLift_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_DeepLift_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_LRP_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_LRP_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_Occlusion_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_Occlusion_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_Lime_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_Lime_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_ShapleyValues_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_FeatureAblation_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_Deconvolution_no_iter_test_h_edge_100,
    'SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_h_edge_100':SimpleCNN_E_MNIST_ShapleyValueSampling_no_iter_test_h_edge_100,

    'vgg16_E_CIFAR10_GradCAM':vgg16_E_CIFAR10_GradCAM,
    'vgg16_E_CIFAR10_GradCAM_on_map':vgg16_E_CIFAR10_GradCAM_on_map,
    'vgg16_E_CIFAR10_Random':vgg16_E_CIFAR10_Random,
    'vgg16_E_CIFAR10_GradCAM_progressive':vgg16_E_CIFAR10_GradCAM_progressive,
    'vgg16_E_CIFAR10_Random_test':vgg16_E_CIFAR10_Random_test,
    'vgg16_E_Food101_Random':vgg16_E_Food101_Random,
    'vgg16_E_CIFAR10_Random_test_h_edge':vgg16_E_CIFAR10_Random_test_h_edge,
    
    'vgg16_E_CIFAR10_Random_test_boot':vgg16_E_CIFAR10_Random_test_boot,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_boot':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_boot,

    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_basemap':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_basemap,
    'vgg16_E_CIFAR10_InputXGradient_no_iter_test_basemap':vgg16_E_CIFAR10_InputXGradient_no_iter_test_basemap,
    'vgg16_E_CIFAR10_Saliency_no_iter_test_basemap':vgg16_E_CIFAR10_Saliency_no_iter_test_basemap,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_basemap':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_basemap,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_basemap':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_basemap,

    'vgg16_E_CIFAR10_Random_test_use_latest_map':vgg16_E_CIFAR10_Random_test_use_latest_map,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_use_latest_map':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_use_latest_map,
    'vgg16_E_CIFAR10_InputXGradient_no_iter_test_use_latest_map':vgg16_E_CIFAR10_InputXGradient_no_iter_test_use_latest_map,
    'vgg16_E_CIFAR10_Random_test_use_latest_map':vgg16_E_CIFAR10_Random_test_use_latest_map,
    'vgg16_E_CIFAR10_Saliency_no_iter_test_use_latest_map':vgg16_E_CIFAR10_Saliency_no_iter_test_use_latest_map,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_use_latest_map':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_use_latest_map,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_use_latest_map':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_use_latest_map,


    'vgg16_E_CIFAR10_Random_val':vgg16_E_CIFAR10_Random_val,
    'vgg16_E_CIFAR10_InputXGradient_no_iter_val':vgg16_E_CIFAR10_InputXGradient_no_iter_val,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_val':vgg16_E_CIFAR10_GuidedGradCam_no_iter_val,
    'vgg16_E_CIFAR10_Saliency_no_iter_val':vgg16_E_CIFAR10_Saliency_no_iter_val,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_val':vgg16_E_CIFAR10_IntegratedGradients_no_iter_val,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_val':vgg16_E_CIFAR10_DeepLiftShap_no_iter_val,

    'vgg16_E_CIFAR10_InputXGradient_no_iter_test':vgg16_E_CIFAR10_InputXGradient_no_iter_test,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test,
    'vgg16_E_CIFAR10_Saliency_no_iter_test':vgg16_E_CIFAR10_Saliency_no_iter_test,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test,
    

    'vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge':vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge,
    'vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge':vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge,

    'vgg16_E_CIFAR10_Random_test_100':vgg16_E_CIFAR10_Random_test_100,
    'vgg16_E_CIFAR10_InputXGradient_no_iter_test_100':vgg16_E_CIFAR10_InputXGradient_no_iter_test_100,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_100':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_100,
    'vgg16_E_CIFAR10_Saliency_no_iter_test_100':vgg16_E_CIFAR10_Saliency_no_iter_test_100,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_100,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_100':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_100,

    'vgg16_E_CIFAR10_GradientShap_no_iter_test_100':vgg16_E_CIFAR10_GradientShap_no_iter_test_100,
    'vgg16_E_CIFAR10_KernelShap_no_iter_test_100':vgg16_E_CIFAR10_KernelShap_no_iter_test_100,
    'vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100':vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_100,
    'vgg16_E_CIFAR10_DeepLift_no_iter_test_100':vgg16_E_CIFAR10_DeepLift_no_iter_test_100,
    'vgg16_E_CIFAR10_LRP_no_iter_test_100':vgg16_E_CIFAR10_LRP_no_iter_test_100,
    'vgg16_E_CIFAR10_Occlusion_no_iter_test_100':vgg16_E_CIFAR10_Occlusion_no_iter_test_100,
    'vgg16_E_CIFAR10_Lime_no_iter_test_100':vgg16_E_CIFAR10_Lime_no_iter_test_100,
    'vgg16_E_CIFAR10_Deconvolution_no_iter_test_100':vgg16_E_CIFAR10_Deconvolution_no_iter_test_100,
    


    'vgg16_E_CIFAR10_InputXGradient_test_100':vgg16_E_CIFAR10_InputXGradient_test_100,
    'vgg16_E_CIFAR10_GuidedGradCam_test_100':vgg16_E_CIFAR10_GuidedGradCam_test_100,
    'vgg16_E_CIFAR10_Saliency_test_100':vgg16_E_CIFAR10_Saliency_test_100,
    'vgg16_E_CIFAR10_IntegratedGradients_test_100':vgg16_E_CIFAR10_IntegratedGradients_test_100,
    'vgg16_E_CIFAR10_DeepLiftShap_test_100':vgg16_E_CIFAR10_DeepLiftShap_test_100,


    'vgg16_E_CIFAR10_Random_test_h_edge_100':vgg16_E_CIFAR10_Random_test_h_edge_100,
    'vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100':vgg16_E_CIFAR10_InputXGradient_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge_100':vgg16_E_CIFAR10_GuidedGradCam_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100':vgg16_E_CIFAR10_Saliency_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100':vgg16_E_CIFAR10_IntegratedGradients_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge_100':vgg16_E_CIFAR10_DeepLiftShap_no_iter_test_h_edge_100,

    'vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100':vgg16_E_CIFAR10_GradientShap_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_KernelShap_no_iter_test_h_edge_100':vgg16_E_CIFAR10_KernelShap_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100':vgg16_E_CIFAR10_GuidedBackprop_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_DeepLift_no_iter_test_h_edge_100':vgg16_E_CIFAR10_DeepLift_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_LRP_no_iter_test_h_edge_100':vgg16_E_CIFAR10_LRP_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_Occlusion_no_iter_test_h_edge_100':vgg16_E_CIFAR10_Occlusion_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_Lime_no_iter_test_h_edge_100':vgg16_E_CIFAR10_Lime_no_iter_test_h_edge_100,
    'vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100':vgg16_E_CIFAR10_Deconvolution_no_iter_test_h_edge_100,
    
    'ResNet50_E_Food101_Random':ResNet50_E_Food101_Random,
    'ResNet50_E_Food101_GradCAM':ResNet50_E_Food101_GradCAM,
    'ResNet50_E_Food101_GradCAM_test':ResNet50_E_Food101_GradCAM_test,
    'ResNet50_E_Food101_Random_test':ResNet50_E_Food101_Random_test,
    
    'ResNet50_E_ImageNet_Random_val_100':ResNet50_E_ImageNet_Random_val_100,
    'ResNet50_E_ImageNet_GuidedGradCam_val_100':ResNet50_E_ImageNet_GuidedGradCam_val_100,
    'ResNet50_E_ImageNet_DeepLiftShap_val_100':ResNet50_E_ImageNet_DeepLiftShap_val_100,
    'ResNet50_E_ImageNet_InputXGradient_val_100':ResNet50_E_ImageNet_InputXGradient_val_100,
    'ResNet50_E_ImageNet_Saliency_val_100':ResNet50_E_ImageNet_Saliency_val_100,
    'ResNet50_E_ImageNet_IntegratedGradients_val_100':ResNet50_E_ImageNet_IntegratedGradients_val_100,

    'ResNet50_E_ImageNet_GradientShap_val_100':ResNet50_E_ImageNet_GradientShap_val_100,
    'ResNet50_E_ImageNet_KernelShap_val_100':ResNet50_E_ImageNet_KernelShap_val_100,
    'ResNet50_E_ImageNet_GuidedBackprop_val_100':ResNet50_E_ImageNet_GuidedBackprop_val_100,
    'ResNet50_E_ImageNet_DeepLift_val_100':ResNet50_E_ImageNet_DeepLift_val_100,
    'ResNet50_E_ImageNet_LRP_val_100':ResNet50_E_ImageNet_LRP_val_100,
    'ResNet50_E_ImageNet_Lime_val_100':ResNet50_E_ImageNet_Lime_val_100,
    'ResNet50_E_ImageNet_Deconvolution_val_100':ResNet50_E_ImageNet_Deconvolution_val_100,


    
    'ResNet50_E_ImageNet_Random_val_h_edge_100':ResNet50_E_ImageNet_Random_val_h_edge_100,
    'ResNet50_E_ImageNet_GuidedGradCam_val_h_edge_100':ResNet50_E_ImageNet_GuidedGradCam_val_h_edge_100,
    'ResNet50_E_ImageNet_DeepLiftShap_val_h_edge_100':ResNet50_E_ImageNet_DeepLiftShap_val_h_edge_100,
    'ResNet50_E_ImageNet_InputXGradient_val_h_edge_100':ResNet50_E_ImageNet_InputXGradient_val_h_edge_100,
    'ResNet50_E_ImageNet_Saliency_val_h_edge_100':ResNet50_E_ImageNet_Saliency_val_h_edge_100,
    'ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100':ResNet50_E_ImageNet_IntegratedGradients_val_h_edge_100,

    'ResNet50_E_ImageNet_GradientShap_val_h_edge_100':ResNet50_E_ImageNet_GradientShap_val_h_edge_100,
    'ResNet50_E_ImageNet_KernelShap_val_h_edge_100':ResNet50_E_ImageNet_KernelShap_val_h_edge_100,
    'ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100':ResNet50_E_ImageNet_GuidedBackprop_val_h_edge_100,
    'ResNet50_E_ImageNet_DeepLift_val_h_edge_100':ResNet50_E_ImageNet_DeepLift_val_h_edge_100,
    'ResNet50_E_ImageNet_LRP_val_h_edge_100':ResNet50_E_ImageNet_LRP_val_h_edge_100,
    'ResNet50_E_ImageNet_Lime_val_h_edge_100':ResNet50_E_ImageNet_Lime_val_h_edge_100,
    'ResNet50_E_ImageNet_Deconvolution_val_h_edge_100':ResNet50_E_ImageNet_Deconvolution_val_h_edge_100,


    
}

def get_config(config_name):
    return CONFIGS[config_name]


if __name__=='__main__':
    print(vgg16_E_CIFAR10.model_name)
    print(vgg16_E_CIFAR10_Random.model_name)