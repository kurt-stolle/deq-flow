#!/bin/bash

python -m deqflow2 --total_run 1 --start_run 1 --name deq-flow-H-naive-120k-C-36-1-1 \
    --stage chairs --validation chairs kitti \
    --gpus 0 1 2 --num_steps 120000 --eval_interval 20000 \
    --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --f_thres 36 --f_solver naive_solver \
    --n_losses 1 --phantom_grad 1 \
    --huge --wnorm

python -m deqflow2 --total_run 1 --start_run 1 --name deq-flow-H-naive-120k-T-40-1-1 \
    --stage things --validation sintel kitti --restore_name deq-flow-H-naive-120k-C-36-1-1 \
    --gpus 0 1 2 --num_steps 120000 \
    --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --f_thres 40 --f_solver naive_solver \
    --n_losses 1 --phantom_grad 1 \
    --huge --wnorm --all_grad 

