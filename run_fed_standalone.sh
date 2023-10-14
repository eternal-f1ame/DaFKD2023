#!/usr/bin/env bash

GPU=0

CLIENT_NUM=5

WORKER_NUM=2

BATCH_SIZE=32

ALPHA=1

DATASET=fashion_mnist
DISTILLATION_DATASET=mnist

MODEL=DaFKD

DISTRIBUTION=hetero

ROUND=30

CLIENT_OPTIMIZER=adam

EPOCH=5

GAN_EPOCH=5

D_EPOCH=1

ED_EPOCH=30
NOISE_DIMENSION=100

TEMPERATURE=20.0
LR=0.0001
ES_LR=0.0001

AGGREGATION=FedDF
CI=0

BASELINE="DaFKD"



CUDA_VISIBLE_DEVICES=0 python main_fed.py --gpu 0 --dataset fashion_mnist --model DaFKD --partition_method hetero --alpha 1 --aggregation_method FedDF --client_num_in_total 5 --client_num_per_round 2 --comm_round 25 --epochs 5 --client_optimizer adam --batch_size 32 --d_epoch 1 --ed_epoch 20 --gan_epoch 5 --noise_dimension 100 --temperature 20.0 --lr 0.0001 --es_lr 0.0001 --ci 0 --baseline "DaFKD" --distillation_dataset mnist
