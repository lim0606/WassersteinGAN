#!/bin/bash

### mnist
# exp1: DCGAN
rm logs/mnist_dcgan*.pkl
grep -r "Loss_D_fake" logs/mnist_dcgan_20170208-151517-402866840.log > logs/mnist_dcgan.log
python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/mnist_dcgan --data dcgan mnist_dcgan.log
eog logs/mnist_dcgan_mdisc_medfilt_loss.png


#### celeba 
## exp1: DCGAN
#rm logs/celeba_dcgan*.pkl
#grep -r "Loss_D_fake" logs/celeba_dcgan_20170208-123330-152273612.log > logs/celeba_dcgan.log
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/celeba_dcgan --data dcgan celeba_dcgan.log
#eog logs/celeba_dcgan_mdisc_medfilt_loss.png
#
#
#### cifar10 
## exp1: DCGAN
#rm logs/cifar10_dcgan*.pkl
#grep -r "Loss_D_fake" logs/xxxxxxxxxx.log > logs/cifar10_dcgan.log
#python plot_log.py -a 500 /home/jaehyun/github/WassersteinGAN/logs logs/cifar10_dcgan --data dcgan cifar10_dcgan.log
#eog logs/cifar10_dcgan_mdisc_loss.png
#
## exp2: MLP 
#grep -r "Loss_D_fake" logs/xxxxxxxxxx.log > logs/cifar10_mlp.log
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/cifar10_mlp --data mlp cifar10_mlp.log
#
## dcgan vs. mlp 
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/cifar10_dcgan_vs_mlp --data dcgan cifar10_dcgan.log --data mlp cifar10_mlp.log
#
#
#### lsun 
## exp1: DCGAN
#rm logs/lsun_dcgan*.pkl
#grep -r "Loss_D_fake" logs/xxxxxxxxxx.log > logs/lsun_dcgan.log
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/lsun_dcgan --data dcgan lsun_dcgan.log
#eog logs/lsun_dcgan_mdisc_medfilt_loss.png
#
## exp2: MLP 
#grep -r "Loss_D_fake" logs/xxxxxxxxxx.log > logs/cifar10_mlp.log
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/cifar10_mlp --data mlp cifar10_mlp.log
#
## dcgan vs. mlp
#python plot_log.py -a 1 /home/jaehyun/github/WassersteinGAN/logs logs/cifar10_dcgan_vs_mlp --data dcgan cifar10_dcgan.log --data mlp cifar10_mlp.log
