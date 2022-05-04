# Reduce the Fidelity Inconsistency in Saliency Metric

This is the project for cse586

please use conda and `environment.yml` to setup the environment.

## Pre-trained Weights

The weights trained on MNIST is [here](https://drive.google.com/drive/folders/1nV-0COQ8lJOeSg8R7vb_2CA8_Lh5o-rP?usp=sharing)
Please download `SimpleCNN_E_MNIST.pt` and put it under `weights`. The weights for other models will be download automatically.

## Get the perturbed images and output

Note that all the experiments are commented out in `run.py`. Please select the corresponding experiment name to run certain experiments. The following is an example experiment.

```sh
  python run.py vgg16_E_CIFAR10_InputXGradient_no_iter_test_100
```

You can then use the following command with corresponding argument (check `generate_results.py` for the avaliable args) names to produce the numbers in the paper.

```sh
  python generate_results.py MNIST_LeRF
```
