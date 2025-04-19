import torch
import random
import os
import numpy as np
from models.MSLE2E6 import URNet
from util.dataset_sorted import RadioCoorDataset
from torch.utils.data import DataLoader

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(seed=42)

# 初始化模型
model = URNet(model_name='M2S', model_type='M2S', img_size=200, patch_size=1,
              in_chans=1, latent_channels=64, out_channels=1, features=[64, 64, 64])

# 训练参数配置（字典）
train_config = {
    'phase1': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 32,
        'freeze': 'model2',
        'save_path': 'model_dic/phase1_num',
        'phase':1
    },
    'phase2': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 32,
        'freeze': 'model1',
        'save_path': 'model_dic/phase2_num',
        'phase':2
    },
    'phase3': {
        'epochs': 10,
        'lr': 1e-4,
        'batch_size': 16,
        'freeze': None,
        'save_path': 'model_dic/phase3_num',
        'phase':3
    }
}
def create_loaders(train_path, val_path, test_path, batch_size):
    """为当前阶段生成数据加载器"""
    train_dataset = RadioCoorDataset(train_path)
    val_dataset = RadioCoorDataset(val_path)
    test_dataset = RadioCoorDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def train_phase(model, phase, config, train_loader, val_loader, device='cuda'):
    """通用阶段训练函数"""

    model.to(device)

    # 冻结或解冻模型
    if config['freeze'] == 'model1':
        for param in model.model1.parameters():
            param.requires_grad = False
        for param in model.model2.parameters():
            param.requires_grad = True
    elif config['freeze'] == 'model2':
        for param in model.model1.parameters():
            param.requires_grad = True
        for param in model.model2.parameters():
            param.requires_grad = False
    else:  # 联合训练
        for param in model.parameters():
            param.requires_grad = True

    # 优化器 & 学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters() if config['freeze'] is None else
        (model.model1.parameters() if config['freeze'] == 'model2' else model.model2.parameters()),
        lr=config['lr'], weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 模型训练
    model.fit_wandb(
        train_loader, val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        phase=phase,
        project_name=f'paper2-e2e',
        run_name = f'e2e',
        epochs=config['epochs'],
        save_model_dir=config['save_path']
    )

    # 保存模型
    save_path = os.path.join(config['save_path'], f'final_weights_phase{phase}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至: {save_path}")

def train_pipeline(model, device='cuda'):
    model.to(device)

    # 加载数据集路径

    train_data_path = 'C:/dataset/MSL-e2e-number/RadioCoorDataset_Train'
    val_data_path = 'C:/dataset/MSL-e2e-number/RadioCoorDataset_Val'

    test_data_path = 'C:/dataset/MSL-e2e-number/RadioCoorDataset_Test'

    # # ✅ 阶段1：训练模型1
    # print("\nPhase 1: Training Model1")
    # loaders = create_loaders(train_data_path, val_data_path, test_data_path, train_config['phase1']['batch_size'])
    # train_phase(model, phase=1, config=train_config['phase1'], train_loader=loaders[0], val_loader=loaders[1], device=device)

    # ✅ 阶段2：训练模型2
    print("\nPhase 2: Training Model2")
    model.load_state_dict(torch.load(train_config['phase1']['save_path'] + '/best_model.pth'))
    loaders = create_loaders(train_data_path, val_data_path, test_data_path, train_config['phase2']['batch_size'])
    train_phase(model, phase=2, config=train_config['phase2'], train_loader=loaders[0], val_loader=loaders[1], device=device)

    # ✅ 阶段3：联合训练
    print("\nPhase 3: Joint Training")
    model.load_state_dict(torch.load(train_config['phase2']['save_path'] + '/final_weights_phase2.pth'))
    loaders = create_loaders(train_data_path, val_data_path, test_data_path, train_config['phase3']['batch_size'])
    train_phase(model, phase=3, config=train_config['phase3'], train_loader=loaders[0], val_loader=loaders[1], device=device)

    # 最终模型保存
    final_model_path = 'model_dic/final_weights.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

if __name__ == "__main__":
    train_pipeline(model)