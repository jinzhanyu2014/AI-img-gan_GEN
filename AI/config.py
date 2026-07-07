import os
import torch

# ========== 自动获取项目根目录 ==========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 所有路径基于根目录拼接 ==========
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(ROOT_DIR, "samples")

# ========== 自动创建必要目录 ==========
for dir_path in [DATA_DIR, IMAGE_DIR, TEXT_DIR, CHECKPOINT_DIR, SAMPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ========== 训练配置 ==========
IMAGE_SIZE = 128
BATCH_SIZE = 8
LATENT_DIM = 100
EPOCHS = 200
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

# ========== 加权采样配置 ==========
HOT_POWER = 1.5          # 热度放大系数，越大越偏向高分图片

# ========== 设备 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"