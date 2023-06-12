# Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection

This is the source code for ICML 2023 paper of Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection by Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon, Robert Nowak, and Yixuan Li.


# Dataset Preparation

Download the data in the folder

```
./data
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




# Datasets

Here are links for the less common outlier datasets used in the paper: [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html), [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz),
and [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy).


# Citation

If you use our codebase, please cite our work:

```
@inproceedings{bai2023feed,
      title={Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection}, 
      author={Haoyue Bai and Gregory Canal and Xuefeng Du and Jeongyeol Kwon and Robert D Nowak and Yixuan Li},
      booktitle = {International Conference on Machine Learning},
      year = {2023}
}
```


