#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python test_v1.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_resnet18_21d_3D_length16_stride4_dropout0.2/model_best.pth \
--arch resnet18_21d \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.2 \
--workers 8 \
--image_tmpl 'img_{:05d}.jpg' \
--save_scores ./output/kinetics400_resnet18_21d_3D_length16_stride4_dropout0.2/ \
