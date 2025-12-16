#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
node_num=8  # the num gpu you want to use
batch_size=128  # per gpu (Total batch size = batch_size * node_num)
imagenet_path='/home/guest/workplace/lzy/imagenet'  # Replace it to your own imagenet path 
model='graphconvnet_ti'   # Change to the model you want to train: graphconvnet_ti, graphconvnet_s, etc.

####### Train Command #######
# Added hyperparameters from local run.sh:
# --opt adamw, --sched cosine, --mixup, --cutmix, --aa, --drop-path, etc.

python -u -m torch.distributed.launch --master_port 22577 --nproc_per_node=${node_num} train.py \
  --data ${imagenet_path} \
  --model ${model} \
  --batch-size ${batch_size} \
  --lr 2e-3 \
  --weight-decay .05 \
  --opt adamw \
  --opt-eps 1e-8 \
  --sched cosine \
  --epochs 300 \
  --warmup-epochs 20 \
  --warmup-lr 1e-6 \
  --mixup .8 \
  --cutmix 1.0 \
  --aa rand-m9-mstd0.5-inc1 \
  --color-jitter 0.4 \
  --repeated-aug \
  --remode pixel \
  --reprob 0.25 \
  --amp \
  --drop 0 \
  --drop-path .1 \
  -j 8 \
  --output ./output/${model}_${batch_size}x${node_num} \
  >./${model}_${batch_size}x${node_num}.log 2>&1 &

###### resume #######
# add " --resume /your/own/checkpoint/path " to the above train command

### Eval Command ##### 
# python train.py --data ${imagenet_path} --model=${model} -b=${batch_size} --pretrain_path /path/to/pretrained/model/ --evaluate
