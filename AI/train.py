import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model import Generator, Discriminator
from dataset import get_weighted_dataloader
from utils import save_checkpoint, load_checkpoint
import config


def train():
    print(f"🚀 开始训练，设备：{config.DEVICE}")
    print(f"📂 数据目录：{config.DATA_DIR}")

    # 加载数据
    dataloader = get_weighted_dataloader(
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        hot_power=config.HOT_POWER
    )

    # 初始化模型
    G = Generator(config.LATENT_DIM, config.IMAGE_SIZE).to(config.DEVICE)
    D = Discriminator(config.IMAGE_SIZE).to(config.DEVICE)

    optim_G = torch.optim.Adam(G.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    optim_D = torch.optim.Adam(D.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    # 加载断点
    g_checkpoint = os.path.join(config.CHECKPOINT_DIR, "latest_G.pth")
    d_checkpoint = os.path.join(config.CHECKPOINT_DIR, "latest_D.pth")

    start_epoch, _ = load_checkpoint(g_checkpoint, G, optim_G)
    load_checkpoint(d_checkpoint, D, optim_D)

    adversarial_loss = nn.BCELoss()

    print(f"🔄 从第 {start_epoch + 1} 轮开始训练")

    for epoch in range(start_epoch, config.EPOCHS):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(config.DEVICE)

            real = torch.ones(batch_size, 1).to(config.DEVICE)
            fake = torch.zeros(batch_size, 1).to(config.DEVICE)

            # ---------- 训练判别器 ----------
            optim_D.zero_grad()
            z = torch.randn(batch_size, config.LATENT_DIM).to(config.DEVICE)
            fake_imgs = G(z)
            loss_D = (adversarial_loss(D(real_imgs), real) +
                      adversarial_loss(D(fake_imgs.detach()), fake)) / 2
            loss_D.backward()
            optim_D.step()

            # ---------- 训练生成器 ----------
            optim_G.zero_grad()
            z = torch.randn(batch_size, config.LATENT_DIM).to(config.DEVICE)
            gen_imgs = G(z)
            loss_G = adversarial_loss(D(gen_imgs), real)
            loss_G.backward()
            optim_G.step()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{config.EPOCHS} | Step {i} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

        # 保存样本
        save_image(gen_imgs, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch+1}.png"), nrow=4, normalize=True)
        save_checkpoint(g_checkpoint, G, optim_G, epoch, loss_G.item())
        save_checkpoint(d_checkpoint, D, optim_D, epoch, loss_D.item())
        print(f"✅ Epoch {epoch+1} 完成，已保存断点及样本")

    print("🎉 训练完成！")


if __name__ == "__main__":
    train()