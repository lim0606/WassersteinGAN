#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs

# mnist with DCGAN 
export CUDA_VISIBLE_DEVICES=1
python main.py --cuda --dataset mnist --dataroot /media/data0/image/mnist --imageSize 32 --nc 1 --niter 5000 --nsave 10 --experiment samples/mnist_dcgan 2>&1 | tee logs/mnist_dcgan_$filename


## celeba with DCGAN 
#export CUDA_VISIBLE_DEVICES=1
#python main.py --cuda --dataset folder --dataroot /media/data0/image/celeba --loadSize 96 --niter 5000 --experiment samples/celeba_dcgan 2>&1 | tee logs/celeba_dcgan_$filename
#
#
## cifar10 with DCGAN 
#export CUDA_VISIBLE_DEVICES=0
#python main.py --cuda --dataset cifar10 --dataroot /media/data0/image/cifar10 --niter 5000 --experiment samples/cifar10_dcgan 2>&1 | tee logs/cifar10_dcgan_$filename
#
## cifar10 with MLP 
#export CUDA_VISIBLE_DEVICES=1
#python main.py --cuda --mlp_G --ngf 512 --dataset cifar10 --dataroot /media/data0/image/cifar10 --niter 5000 --experiment samples/cifar10_mlp 2>&1 | tee logs/cifar10_mlp_$filename
#
#
## lsun with DCGAN
#export CUDA_VISIBLE_DEVICES=0
#python main.py --cuda --dataset lsun --dataroot /media/data0/image/lsun --experiment samples/lsun_dcgan 2>&1 | tee logs/lsun_dcgan_$filename
