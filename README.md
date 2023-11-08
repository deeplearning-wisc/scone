# Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection

This is the source code for ICML 2023 paper of [Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection](https://proceedings.mlr.press/v202/bai23a/bai23a.pdf) by Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon, Robert Nowak, and Yixuan Li.

# Abstract

Modern machine learning models deployed in the wild can encounter both covariate and semantic shifts, giving rise to the problems of out-of-distribution (OOD) generalization and OOD detection respectively. While both problems have received significant research attention lately, they have been pursued independently. This may not be surprising, since the two tasks have seemingly conflicting goals. This paper provides a new unified approach that is capable of simultaneously generalizing to covariate shifts while robustly detecting semantic shifts. We propose a margin-based learning framework that exploits freely available unlabeled data in the wild that captures the environmental test-time OOD distributions under both covariate and semantic shifts. We show both empirically and theoretically that the proposed margin constraint is the key to achieving both OOD generalization and detection.  Extensive experiments show the superiority of our framework, outperforming competitive baselines that specialize in either OOD generalization or OOD detection.

# Preliminaries

This is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)==1.8.1
* [torchvision](https://pypi.org/project/torchvision/)==0.9.1
* [numpy](http://www.numpy.org/)==1.20.3
* [sklearn](https://scikit-learn.org/stable/)==0.24.2
* [wandb](https://pypi.org/project/wandb/)

# Dataset Preparation

Download the data in the folder

```
./data
```

Here are links for the less common OOD datasets used in the paper: 
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html), 
[LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz),
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd data/LSUN
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

The corrupted CIFAR-10 dataset can be downloaded via the link:
```
wget https://drive.google.com/drive/u/0/folders/1JcI8UMBpdMffzCe-dqrzXA9bSaEGItzo
```


For large-scale experiments, we use iNaturalist as the semantic OOD dataset. We have sampled 10,000 images from the selected concepts for iNaturalist,
which can be downloaded via the following link:
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
```



# Pretrained models

You can find the pretrained models in 

```
./CIFAR/snapshots/pretrained
```


# Demo

To run the code, execute 

```
bash run.sh score in_distribution aux_distribution test_distribution 
```

For example, to run woods on cifar10 using svhn as the mixture distribution and the test_distribution, execute

```
bash run.sh scone cifar10 svhn svhn
```

pi_1 is set to 0.5 and pi_2 is set to 0.1 as default. See the run.sh for more details and options. 

# Main Files

* ```CIFAR/train.py``` contains the main code used to train model(s) under our framework.
* ```CIFAR/make_datasets.py``` contains the code for reading datasets into PyTorch.
* ```CIFAR/plot_results.py``` contains code for loading and analyzing experimental results.
* ```CIFAR/test.py``` contains code for testing experimental results in OOD setting.



# Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{bai2023feed,
      title={Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection}, 
      author={Haoyue Bai and Gregory Canal and Xuefeng Du and Jeongyeol Kwon and Robert D Nowak and Yixuan Li},
      booktitle = {International Conference on Machine Learning},
      year = {2023}
}
```
Our codebase borrows from the following:
```
@inproceedings{katz2022training,
  title={Training Ood Detectors in Their Natural Habitats},
  author={Katz-Samuels, Julian and Nakhleh, Julia B and Nowak, Robert and Li, Yixuan},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

