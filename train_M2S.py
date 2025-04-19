import torch
import random
import os
import numpy as np
from models.Multi2Single import URNet
from util.dataset_M2S import RadioCoorDataset
from torch.utils.data import DataLoader


def seed_torch(seed=12):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

model = URNet(model_name='M2S', model_type='M2S', img_size=200, patch_size=1, in_chans=1, latent_channels=64,
             out_channels=1, features=[64, 64, 64])

train_data_path = 'C:/dataset/Vary_source2/RadioCoorDataset_Train'
val_data_path = 'C:/dataset/Vary_source2/RadioCoorDataset_Val'
test_data_path = 'C:/dataset/Vary_source2/RadioCoorDataset_Test'
json_data_path = 'D:/paper code/dataset/MultiSourceDataset/antenna_sorted'

# 创建数据集对象
train_dataset = RadioCoorDataset(train_data_path, json_data_path)
val_dataset = RadioCoorDataset(val_data_path, json_data_path)
test_dataset = RadioCoorDataset(test_data_path, json_data_path)
# 定义批量大小（batch size）
batch_size = 32
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
min_samples = 400
max_samples = 400
dB_max = -68
dB_min = -130
free_space_only = False
pre_sampled = True
# 训练模型
model.to('cuda')

model.fit_wandb(train_loader, val_loader, optimizer, scheduler,min_samples, max_samples, pre_sampled,project_name = 'SourceLocalization-main',
                run_name="M2SGLOBAL",dB_max=-47.84, dB_min=-147,epochs=10, save_model_epochs=50,save_model_dir='model_dic/M2SGLOBAL')

