import argparse
import torch
import numpy as np
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='./sarbest.pt', help='./sarbest.pt')
#     parser.add_argument('--ToFP16', action='store_true', help='convert model from FP32 to FP16')
#     opt = parser.parse_args()
#
#     # Load pytorch model
#     # torch.load读取的pytorch模型.pt文件是一个字典，包含五个key：(我们只需要best_fitness和model)
#     # epoch——当前模型对应epoch数；
#     # best_fitness——模型性能加权指标，是一个值；
#     # training_results——每个epoch的mAP、loss等信息；
#     # model——保存的模型
#     # optimizer——优化信息，一般占.pt文件一半大小
#     model = torch.load(opt.weights, map_location=torch.device('cpu'))
#
#     # 读取模型网络
#     net = model['model']
#
#     # 把模型从单精度FP32转为半精度FP16
#     if opt.ToFP16:
#         net.half()
#
#     # 只保留有用信息
#     # 预训练权重模型默认epoch=-1，不保存training_results，不保存optimizer，相当于只保存了模型和权重
#     ckpt = {'epoch': -1,
#             'best_fitness': model['best_fitness'],
#             'training_results': None,
#             'model': net,
#             'optimizer': None}
#
#     # 保存模型
#     torch.save(ckpt, 'test-slim.pt')
#
#     print('=========DONE=========')
model.0.cv1.conv.weight
model.0.cv1.bn.weight
model.0.cv1.bn.bias
model.0.cv2.conv.weight
freezing model.0.cv2.conv.weight
model.0.cv2.bn.weight
freezing model.0.cv2.bn.weight
model.0.cv2.bn.bias
freezing model.0.cv2.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
model.2.cv1.bn.bias
model.2.cv2.conv.weight
model.2.cv2.bn.weight
model.2.cv2.bn.bias
model.2.cv3.conv.weight
model.2.cv3.bn.weight
model.2.cv3.bn.bias
model.2.m.0.cv1.conv.weight
model.2.m.0.cv1.bn.weight
model.2.m.0.cv1.bn.bias
model.2.m.0.cv2.conv.weight
model.2.m.0.cv2.bn.weight
model.2.m.0.cv2.bn.bias
model.3.conv.weight
model.3.bn.weight
model.3.bn.bias
model.4.cv1.conv.weight
model.4.cv1.bn.weight
model.4.cv1.bn.bias
model.4.cv2.conv.weight
model.4.cv2.bn.weight
model.4.cv2.bn.bias
model.4.cv3.conv.weight
model.4.cv3.bn.weight
model.4.cv3.bn.bias
model.4.m.0.cv1.conv.weight
model.4.m.0.cv1.bn.weight
model.4.m.0.cv1.bn.bias
model.4.m.0.cv2.conv.weight
model.4.m.0.cv2.bn.weight
model.4.m.0.cv2.bn.bias
model.4.m.1.cv1.conv.weight
model.4.m.1.cv1.bn.weight
model.4.m.1.cv1.bn.bias
model.4.m.1.cv2.conv.weight
model.4.m.1.cv2.bn.weight
model.4.m.1.cv2.bn.bias
model.5.conv.weight
model.5.bn.weight
model.5.bn.bias
model.6.cv1.conv.weight
model.6.cv1.bn.weight
model.6.cv1.bn.bias
model.6.cv2.conv.weight
model.6.cv2.bn.weight
model.6.cv2.bn.bias
model.6.cv3.conv.weight
model.6.cv3.bn.weight
model.6.cv3.bn.bias
model.6.m.0.cv1.conv.weight
model.6.m.0.cv1.bn.weight
model.6.m.0.cv1.bn.bias
model.6.m.0.cv2.conv.weight
model.6.m.0.cv2.bn.weight
model.6.m.0.cv2.bn.bias
model.6.m.1.cv1.conv.weight
model.6.m.1.cv1.bn.weight
model.6.m.1.cv1.bn.bias
model.6.m.1.cv2.conv.weight
model.6.m.1.cv2.bn.weight
model.6.m.1.cv2.bn.bias
model.6.m.2.cv1.conv.weight
model.6.m.2.cv1.bn.weight
model.6.m.2.cv1.bn.bias
model.6.m.2.cv2.conv.weight
model.6.m.2.cv2.bn.weight
model.6.m.2.cv2.bn.bias
model.7.conv.weight
model.7.bn.weight
model.7.bn.bias
model.8.cv1.conv.weight
model.8.cv1.bn.weight
model.8.cv1.bn.bias
model.8.cv2.conv.weight
model.8.cv2.bn.weight
model.8.cv2.bn.bias
model.8.cv3.conv.weight
model.8.cv3.bn.weight
model.8.cv3.bn.bias
model.8.m.0.cv1.conv.weight
model.8.m.0.cv1.bn.weight
model.8.m.0.cv1.bn.bias
model.8.m.0.cv2.conv.weight
model.8.m.0.cv2.bn.weight
model.8.m.0.cv2.bn.bias
model.9.cv1.conv.weight
model.9.cv1.bn.weight
model.9.cv1.bn.bias
model.9.cv2.conv.weight
model.9.cv2.bn.weight
model.9.cv2.bn.bias
model.10.conv.weight
model.10.bn.weight
model.10.bn.bias
model.13.cv1.conv.weight
model.13.cv1.bn.weight
model.13.cv1.bn.bias
model.13.cv2.conv.weight
model.13.cv2.bn.weight
model.13.cv2.bn.bias
model.13.cv3.conv.weight
model.13.cv3.bn.weight
model.13.cv3.bn.bias
model.13.m.0.cv1.conv.weight
model.13.m.0.cv1.bn.weight
model.13.m.0.cv1.bn.bias
model.13.m.0.cv2.conv.weight
model.13.m.0.cv2.bn.weight
model.13.m.0.cv2.bn.bias
model.14.conv.weight
model.14.bn.weight
model.14.bn.bias
model.17.cv1.conv.weight
model.17.cv1.bn.weight
model.17.cv1.bn.bias
model.17.cv2.conv.weight
model.17.cv2.bn.weight
model.17.cv2.bn.bias
model.17.cv3.conv.weight
model.17.cv3.bn.weight
model.17.cv3.bn.bias
model.17.m.0.cv1.conv.weight
model.17.m.0.cv1.bn.weight
model.17.m.0.cv1.bn.bias
model.17.m.0.cv2.conv.weight
model.17.m.0.cv2.bn.weight
model.17.m.0.cv2.bn.bias
model.18.conv.weight
model.18.bn.weight
model.18.bn.bias
model.20.cv1.conv.weight
model.20.cv1.bn.weight
model.20.cv1.bn.bias
model.20.cv2.conv.weight
model.20.cv2.bn.weight
model.20.cv2.bn.bias
model.20.cv3.conv.weight
model.20.cv3.bn.weight
model.20.cv3.bn.bias
model.20.m.0.cv1.conv.weight
model.20.m.0.cv1.bn.weight
model.20.m.0.cv1.bn.bias
model.20.m.0.cv2.conv.weight
model.20.m.0.cv2.bn.weight
model.20.m.0.cv2.bn.bias
model.21.conv.weight
model.21.bn.weight
model.21.bn.bias
model.23.cv1.conv.weight
model.23.cv1.bn.weight
model.23.cv1.bn.bias
model.23.cv2.conv.weight
model.23.cv2.bn.weight
model.23.cv2.bn.bias
model.23.cv3.conv.weight
model.23.cv3.bn.weight
model.23.cv3.bn.bias
model.23.m.0.cv1.conv.weight
model.23.m.0.cv1.bn.weight
model.23.m.0.cv1.bn.bias
model.23.m.0.cv2.conv.weight
model.23.m.0.cv2.bn.weight
model.23.m.0.cv2.bn.bias
model.24.m.0.weight
model.24.m.0.bias
model.24.m.1.weight
model.24.m.1.bias
model.24.m.2.weight
model.24.m.2.bias