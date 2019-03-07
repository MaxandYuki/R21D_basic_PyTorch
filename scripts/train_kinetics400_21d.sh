#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_21d.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet18_21d \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--num_segments 1 \
--epochs 95 \
--batch-size 128 \
--lr 0.1 \
--lr_steps 3 56 78 86 90 92 \
--workers 8 \
--before_softmax \
--image_tmpl 'img_{:05d}.jpg' \
--zero_val \
