import os
import torch
import config


def save_checkpoint(filepath, model, optimizer, epoch, loss):
    """保存断点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """加载断点"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0)
    return 0, 0


def get_latest_checkpoint():
    """获取最新的断点文件路径"""
    if not os.path.exists(config.CHECKPOINT_DIR):
        return None, None
    g_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('latest_G')]
    d_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('latest_D')]
    if g_files and d_files:
        return os.path.join(config.CHECKPOINT_DIR, g_files[0]), os.path.join(config.CHECKPOINT_DIR, d_files[0])
    return None, None