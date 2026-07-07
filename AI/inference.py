import os
import torch
from torchvision.utils import save_image

from model import Generator
import config


def generate_image(output_path="generated.png"):
    """生成一张图片"""
    G = Generator(config.LATENT_DIM, config.IMAGE_SIZE).to(config.DEVICE)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_G.pth")
    if not os.path.exists(checkpoint_path):
        print("❌ 未找到训练好的模型，请先运行 train.py 训练")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    G.load_state_dict(checkpoint['model_state_dict'])
    G.eval()

    z = torch.randn(1, config.LATENT_DIM).to(config.DEVICE)
    with torch.no_grad():
        img = G(z)

    save_image(img, output_path, normalize=True)
    print(f"✅ 图片已保存：{output_path}")


if __name__ == "__main__":
    generate_image()