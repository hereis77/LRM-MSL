import math
import torch
import torch.nn as nn
from util.dataset_M2S import RadioCoorDataset
from torch.utils.data import DataLoader
from models.Multi2Single import URNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = URNet(model_name='M2S', model_type='M2S', img_size=200, patch_size=1, in_chans=1, latent_channels=64,
             out_channels=1, features=[64, 64, 64])
model_path = r'D:/paper code/SourceLocalization-main/model_dic/VaryM2S_2/VaryM2S_2/VaryM2S_2/model.pth'  # 模型路径
model.load_state_dict(torch.load(model_path))
model.eval()
# 加载测试数据
test_data_path = 'C:/dataset/Vary_source2/RadioCoorDataset_Test'
json_data_path = 'D:/pythonfile/MyJupyter/Multi-Source Dataset/VaryTxPower/antenna'
test_dataset = RadioCoorDataset(test_data_path,json_data_path)
# print(test_dataset)
print("Number of samples in test dataset:", len(test_dataset))
batch_size =1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
min_samples = 400  # 每个batch的最小样本数
max_samples = 400  # 每个batch的最大样本数

free_space_only = False  # 是否只考虑自由空间
pre_sampled = True  # 是否预先采样
model.to('cuda')
model.evaluate_and_visualize(test_loader, free_space_only=False, pre_sampled=True)

