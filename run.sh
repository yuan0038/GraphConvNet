export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
node_num=8  # the num gpu you want to use
batch_size=128  #per gpu
imagenet_path='/home/guest/workplace/lzy/imagenet'  #Relace it to your own imagenet path 
model='graphconvnet_ti'   # Change to the model you want to train:graphconvnet_ti graphconvnet_s  graphconvnetp_ti graphconvnetp_s

####### Train Command######:Replace  'your/imagenet/root/path' to your own ImageNet path
python -u -m torch.distributed.launch --master_port 22577 --nproc_per_node=${node_num} train.py  --data ${imagenet_path} --model  ${model}  --lr 2e-3   --output ./output/${model}_${batch_size}x${node_num}   >./${model}_${batch_size}x${node_num}.log 2>&1 &


###### resume    #######
#add  " --resume /your/own/checkpoint/path "   to the above train command

### Eval Command ##### 1. Replace your/imagenet/root/path to your own ImageNet path 2.Replace /path/to/pretrained/model/ to the pretrained ckpt path
# python train.py --data ${imagenet_path} --model=${model} -b=${batch_size} --pretrain_path /path/to/pretrained/model/ --evaluate
