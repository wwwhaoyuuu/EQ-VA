OMP_NUM_THREADS=1 

export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6

torchrun --nnodes=1 --master_port 10006 --nproc_per_node=6 PI/shape[bs,n_channels,n_second,de_features]/run_Comparative_EXP.py \
        --world_size 6 \
        --model EEGNet_classifier \
        --num_chs 62 \
        --emotion negative \
        --codebook_dim 5 \
        --batch_size 128 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 100 \
        --save_ckpt_freq 10 \
        --gradient_accumulation_steps 1 \
        --seed 0 \