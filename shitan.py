import torch

path1='yolov5s_sar.pt'
pretrain_modelone=torch.load(path1,map_location='cpu')
print(pretrain_modelone)