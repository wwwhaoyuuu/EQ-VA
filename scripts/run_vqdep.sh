#!/bin/bash

# 设置OpenMP线程数
export OMP_NUM_THREADS=32

export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6

# 运行PyTorch分布式训练命令
torchrun --nnodes=1 --master_port 10003 --nproc_per_node=6 PI/shape[bs,n_channels,n_second,de_features]/run_vqdep_training.py \
    --output_dir PI/shape[bs,n_channels,n_second,de_features]/checkpoints/vqdep_neutral_test/ \
    --log_dir PI/shape[bs,n_channels,n_second,de_features]/log/vqdep_neutral_test/ \
    --world_size 6 \
    --emotion neutral \
    --model vqdep_vocab_1k_dim_32 \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 5 \
    --quantize_kmeans_init \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --epochs 101 \
    --save_ckpt_freq 100 \
    --ema_decay 0.99 \
    # --num_chs 62 \

echo "Training script has completed execution."
