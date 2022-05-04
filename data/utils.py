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
    LRP,
    Occlusion, # slow
    Lime,
    ShapleyValues,# not use too slow
    FeatureAblation,# not use no reference
    ShapleyValueSampling,# not use too slow
    Deconvolution)
import numpy as np
import torch

def check_type(cls1,cls2):
    # print(type(cls1),type(cls2))
    return cls1 == cls2

def attribute_image_object(algorithm,alg_args={}):
    # print(alg_args.)
    if check_type(algorithm,GuidedGradCam):
        attr_obj = algorithm(model=alg_args['model'],layer=alg_args['layer'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,target=att_args.target)
    elif check_type(algorithm,LRP):
        attr_obj = algorithm(model=alg_args['model'])
    elif check_type(algorithm,ShapleyValueSampling):
        attr_obj = algorithm(forward_func=alg_args['model'])
    elif check_type(algorithm,ShapleyValues):
        attr_obj = algorithm(forward_func=alg_args['model'])
    elif check_type(algorithm,FeatureAblation):
        attr_obj = algorithm(forward_func=alg_args['model'])
    elif check_type(algorithm,Occlusion):
        attr_obj = algorithm(forward_func=alg_args['model'])
    elif check_type(algorithm,Lime):
        attr_obj = algorithm(forward_func=alg_args['model'])
    elif check_type(algorithm,GradientShap):
        attr_obj = algorithm(forward_func=alg_args['model'],multiply_by_inputs=alg_args['multiply_by_inputs'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,baselines=att_args.baselines,target=att_args.target)
    elif check_type(algorithm,DeepLiftShap):
        attr_obj = algorithm(model=alg_args['model'],multiply_by_inputs=alg_args['multiply_by_inputs'])
    elif check_type(algorithm,DeepLift):
        attr_obj = algorithm(model=alg_args['model'],multiply_by_inputs=alg_args['multiply_by_inputs'])
    elif check_type(algorithm,KernelShap):
        attr_obj = algorithm(forward_func=alg_args['model'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,baselines=att_args.baselines,target=att_args.target)
    elif check_type(algorithm,InputXGradient):
        attr_obj = algorithm(forward_func=alg_args['model'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,target=att_args.target)
    elif check_type(algorithm,Saliency):
        attr_obj = algorithm(forward_func=alg_args['model'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,target=att_args.target)
    elif check_type(algorithm,IntegratedGradients):
        attr_obj = algorithm(forward_func=alg_args['model'],multiply_by_inputs=alg_args['multiply_by_inputs'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,baselines=att_args.baselines,target=att_args.target)
    elif check_type(algorithm,GuidedBackprop):
        attr_obj = algorithm(model=alg_args['model'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,target=att_args.target)
    elif check_type(algorithm,Deconvolution):
        attr_obj = algorithm(model=alg_args['model'])
        # tensor_attributions = attr_obj.attribute(inputs=att_args.inputs,target=att_args.target)
    else:
        raise KeyError(f'The algorithm {algorithm} is not defined')
    return attr_obj


def attribute_image_features(attr_obj,channel_aggregation='abs', att_args={}):
    # net.zero_grad()
    if isinstance(attr_obj,GuidedGradCam):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,LRP):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,ShapleyValues):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,ShapleyValueSampling):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'],n_samples=10)
    elif isinstance(attr_obj,Occlusion):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'], sliding_window_shapes=(1,3,3))
    elif isinstance(attr_obj,FeatureAblation):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,Lime):
        # attr_obj = algorithm(model=alg_args.model,layer=alg_args.layer)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'],n_samples=200)
    elif isinstance(attr_obj,GradientShap):
        # attr_obj = algorithm(forward_func=alg_args.model,multiply_by_inputs=alg_args.multiply_by_inputs)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],baselines=att_args['baselines'],target=att_args['target'])
    elif isinstance(attr_obj,DeepLiftShap):
        # attr_obj = algorithm(forward_func=alg_args.model,multiply_by_inputs=alg_args.multiply_by_inputs)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],baselines=att_args['baselines'],target=att_args['target'])
    elif isinstance(attr_obj,DeepLift):
        # attr_obj = algorithm(forward_func=alg_args.model,multiply_by_inputs=alg_args.multiply_by_inputs)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,KernelShap):
        # attr_obj = algorithm(forward_func=alg_args.model)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,InputXGradient):
        # attr_obj = algorithm(forward_func=alg_args.model)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,Saliency):
        # attr_obj = algorithm(forward_func=alg_args.model)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,IntegratedGradients):
        # attr_obj = algorithm(forward_func=alg_args.model,multiply_by_inputs=alg_args.multiply_by_inputs)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,GuidedBackprop):
        # attr_obj = algorithm(model=alg_args.model)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    elif isinstance(attr_obj,Deconvolution):
        # attr_obj = algorithm(model=alg_args.model)
        tensor_attributions = attr_obj.attribute(inputs=att_args['inputs'],target=att_args['target'])
    else:
        raise KeyError(f'The algorithm {attr_obj} is not defined')
    # print(torch.reshape(tensor_attributions,(-1,tensor_attributions.size(dim=1),tensor_attributions.size(dim=2))).cpu().detach().numpy().shape)
    tensor_attributions = np.transpose(torch.reshape(tensor_attributions,(-1,tensor_attributions.size(dim=2),tensor_attributions.size(dim=3))).cpu().detach().numpy(), (1, 2, 0))
    # tensor_attributions = tensor_attributions.max(2) # channal-wise max
    # print(tensor_attributions.shape,channel_aggregation)
    if channel_aggregation=='abs':
        tensor_attributions = np.abs(tensor_attributions) 
    elif channel_aggregation=='all':
        tensor_attributions = tensor_attributions
    elif channel_aggregation=='pos':
        tensor_attributions = (tensor_attributions > 0) * tensor_attributions
    elif channel_aggregation=='neg':
        tensor_attributions = (tensor_attributions < 0) * tensor_attributions
    else:
        raise KeyError(f'The aggregation {channel_aggregation} is not defined')
    # tensor_attributions = tensor_attributions.max(2) # channal-wise max
    tensor_attributions = tensor_attributions.mean(2) # channal-wise max
    return tensor_attributions