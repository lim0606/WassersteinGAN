#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

mkdir logs

### mnist
# WGAN, DCGAN
export CUDA_VISIBLE_DEVICES=0
python main.py --cuda --dataset mnist --dataroot /media/data0/image/mnist --imageSize 32 --nc 1 \
  --niter 5000 --nsave 10 \
  --clamp \
  --experiment samples/mnist_wgan_dcgan \
  2>&1 | tee logs/mnist_wgan_dcgan_$filename

# GAN, DCGAN
export CUDA_VISIBLE_DEVICES=1
python main.py --cuda --dataset mnist --dataroot /media/data0/image/mnist --imageSize 32 --nc 1 \
  --niter 500 --nsave 10 \
  --mode gan --adam --lrD 0.0002 --lrG 0.0002 \
  --Diters 1 \
  --experiment samples/mnist_gan_dcgan \
  2>&1 | tee logs/mnist_gan_dcgan_$filename


### celeba
# WGAN, DCGAN 
export CUDA_VISIBLE_DEVICES=0
python main.py --cuda --dataset folder --dataroot /media/data0/image/celeba --loadSize 96 \
  --niter 5000 --nsave 10 \
  --clamp \
  --experiment samples/celeba_wgan_dcgan \
  2>&1 | tee logs/celeba_wgan_dcgan_$filename

# GAN, DCGAN
export CUDA_VISIBLE_DEVICES=1
python main.py --cuda --dataset foler --dataroot /media/data0/image/celeba --loadSize 96 \
  --niter 1000 --nsave 10 \
  --mode gan --adam --lrD 0.0002 --lrG 0.0002 \
  --Diters 1 \
  --experiment samples/celeba_gan_dcgan \
  2>&1 | tee logs/celeba_gan_dcgan_$filename


### cifar10
# WGAN, DCGAN 
export CUDA_VISIBLE_DEVICES=0
python main.py --cuda --dataset cifar10 --dataroot /media/data0/image/cifar10 \
  --niter 5000 --nsave 10 \
  --clamp \
  --experiment samples/cifar10_wgan_dcgan \
  2>&1 | tee logs/cifar10_wgan_dcgan_$filename


# WGAN, MLP 
export CUDA_VISIBLE_DEVICES=1
python main.py --cuda --dataset cifar10 --dataroot /media/data0/image/cifar10 \
  --niter 5000 --nsave \
  --clamp \
  --mlp_G --ngf 512 \
  --experiment samples/cifar10_wgan_mlp \
  2>&1 | tee logs/cifar10_wgan_mlp_$filename


### lsun 
# WGAN, DCGAN
export CUDA_VISIBLE_DEVICES=0
python main.py --cuda --dataset lsun --dataroot /media/data0/image/lsun \
  --clamp \
  --experiment samples/lsun_wgan_dcgan \
  2>&1 | tee logs/lsun_wgan_dcgan_$filename
