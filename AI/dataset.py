import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import config


class WeightedImageDataset(Dataset):
    """带权重的图片数据集，重点学习热度高的图片"""
    def __init__(self, image_size=64, hot_power=1.5):
        self.image_dir = config.IMAGE_DIR
        self.metadata_file = config.METADATA_FILE
        self.hot_power = hot_power

        self.metadata = self._load_metadata()
        self.samples, self.weights = self._build_samples()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print(f"✅ 有效样本: {len(self.samples)} 张")

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 如果没有metadata，扫描图片目录
        metadata = []
        for img_file in os.listdir(self.image_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                metadata.append({
                    "image": img_file,
                    "text": "",
                    "hot_score": 1
                })
        return metadata

    def _build_samples(self):
        samples = []
        weights = []

        hot_scores = [item.get('hot_score', 1) for item in self.metadata]
        min_hot = max(min(hot_scores) if hot_scores else 1, 1)
        max_hot = max(hot_scores) if hot_scores else 1

        for item in self.metadata:
            img_path = os.path.join(self.image_dir, item['image'])
            if not os.path.exists(img_path):
                continue

            hot = item.get('hot_score', 1)
            normalized = (hot - min_hot) / (max_hot - min_hot + 1e-6)
            weight = max(normalized ** self.hot_power, 0.1)

            samples.append({
                'img_path': img_path,
                'text': item.get('text', ''),
                'hot_score': hot
            })
            weights.append(weight)

        return samples, weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(self.weights[idx], dtype=torch.float32)


def get_weighted_dataloader(batch_size=16, image_size=64, hot_power=1.5):
    """获取带权重的 DataLoader"""
    dataset = WeightedImageDataset(image_size, hot_power)
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)