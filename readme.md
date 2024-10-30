![image](./fig/grapgconvnet.png)

## Requirements

Pytorch 1.7.0,
timm 0.3.2,
torchprofile 0.0.4,
apex

## Pretrained models

- GraphConvNet

| Model            | Params (M) | FLOPs (B) | Top-1 | BaiduDisk URL                                                |
|------------------|------------|-----------|-------| ------------------------------------------------------------ |
| GraphConvNet-Ti  | 7.7        | 1.3       | 77.1  | [BaiduDisk URL](https://pan.baidu.com/s/1_yCwQnPhneGnho6AaT-cBw?pwd=5eri) |
| GraphConvNet-S   | 24.5       | 4.9       | 82.0  | [BaiduDisk URL](https://pan.baidu.com/s/1EBXv987qj9p5X5_OtCOcDA?pwd=hji9) |


- Pyramid GraphConvNet

| Model                   | Params (M) | FLOPs (B) | Top-1 | BaiduDisk URL                                                             |
|-------------------------|------------|-----------|-------|---------------------------------------------------------------------------|
| Pyramid GraphConvNet-Ti | 11.4       | 1.8       | 80.5  | [BaiduDisk URL](https://pan.baidu.com/s/1nYOAoe8R3jf4KMjIWw-KAA?pwd=tmsb) |
| Pyramid GraphConvNet-S  | 29.2       | 4.9       | 82.4  | [BaiduDisk URL](https://pan.baidu.com/s/1KJnmqEmqiw17zV64qqNQRw?pwd=tvkv) |


 
 


## Train ＆ Evaluation
see  `run.sh`


## Acknowledgement

This repo partially uses code from [vig](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)