import os
import json
import time
import random
import hashlib
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

class SmartCrawler:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        self.base_image_dir = os.path.join(save_dir, "images")
        self.base_text_dir = os.path.join(save_dir, "texts")
        self.metadata_file = os.path.join(save_dir, "metadata.json")

        self.styles = ["赛博朋克", "可爱", "古风"]
        
        self.image_dirs = {}
        self.text_dirs = {}
        for style in self.styles:
            self.image_dirs[style] = os.path.join(self.base_image_dir, style)
            self.text_dirs[style] = os.path.join(self.base_text_dir, style)
            os.makedirs(self.image_dirs[style], exist_ok=True)
            os.makedirs(self.text_dirs[style], exist_ok=True)

        self.metadata = self._load_metadata()
        self.downloaded_hashes = set()
        for item in self.metadata:
            img_path = os.path.join(self.base_image_dir, item['style'], item['image'])
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    self.downloaded_hashes.add(hashlib.md5(f.read()).hexdigest())

        print(f"📂 加载已有数据: {len(self.metadata)} 张图片")
        print(f"🎨 风格: {self.styles}")

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _get_bing_images(self, keyword, max_count=100):
        urls = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/121.0.0.0'
        }
        
        for offset in range(0, max_count, 35):
            url = f'https://www.bing.com/images/search?q={keyword}&first={offset}&count=35&form=QBIR'
            
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                for img_tag in soup.find_all('img'):
                    src = img_tag.get('src') or img_tag.get('data-src')
                    if src and 'http' in src and not src.endswith('.svg'):
                        if 'th?id=' in src or 'https://tse' in src:
                            urls.append(src)
                
                time.sleep(0.5)
            except Exception as e:
                continue
        
        urls = list(set(urls))
        valid_urls = []
        for u in urls:
            if 'http' in u and not u.endswith('.svg'):
                valid_urls.append(u)
        
        return valid_urls[:max_count]

    def _download_image(self, url, style, search_keyword):
        try:
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code != 200:
                return None

            img_data = resp.content
            img_hash = hashlib.md5(img_data).hexdigest()
            if img_hash in self.downloaded_hashes:
                return None
            self.downloaded_hashes.add(img_hash)

            img = Image.open(BytesIO(img_data))
            if img.size[0] < 64 or img.size[1] < 64:
                return None
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_name = f"{int(time.time())}_{random.randint(1000,9999)}.jpg"
            img_path = os.path.join(self.image_dirs[style], img_name)
            img.save(img_path)

            prompt = f"一张{style}风格的动漫头像"
            txt_path = os.path.join(self.text_dirs[style], img_name.replace('.jpg', '.txt'))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            self.metadata.append({
                "image": img_name,
                "style": style,
                "text": prompt,
                "search_keyword": search_keyword,
                "source_url": url
            })

            if len(self.metadata) % 10 == 0:
                self._save_metadata()

            return img_name

        except Exception as e:
            return None

    def crawl_until(self, target_per_style=10000):
        style_keywords = {
            "赛博朋克": ["赛博朋克动漫", "cyberpunk anime", "赛博朋克头像"],
            "可爱": ["可爱动漫", "kawaii anime", "可爱头像"],
            "古风": ["古风动漫", "中国风动漫", "古风头像"]
        }

        print(f"🚀 开始爬取，每种风格目标 {target_per_style} 张")
        print("⚠️  按 Ctrl+C 可随时暂停，进度不丢")

        try:
            for style in self.styles:
                current_count = len([item for item in self.metadata if item['style'] == style])
                print(f"\n📂 风格 [{style}] 已有 {current_count} 张，目标 {target_per_style} 张")

                while current_count < target_per_style:
                    keywords = style_keywords[style]
                    search_word = keywords[current_count % len(keywords)]
                    
                    print(f"🕷️  [{style}] 正在必应搜索: {search_word}")
                    urls = self._get_bing_images(search_word, max_count=100)

                    if not urls:
                        print(f"⚠️ 搜索词 {search_word} 没拿到图，换下一个")
                        continue

                    downloaded = 0
                    for url in urls:
                        if current_count >= target_per_style:
                            break
                        result = self._download_image(url, style, search_word)
                        if result:
                            downloaded += 1
                            current_count += 1
                            if downloaded % 5 == 0:
                                print(f"   [{style}] 进度: {current_count}/{target_per_style}")

                    print(f"   [{style}] 本轮下载 {downloaded} 张，总进度 {current_count}/{target_per_style}")
                    time.sleep(random.randint(2, 5))

                print(f"✅ [{style}] 已完成！共 {current_count} 张")

        except KeyboardInterrupt:
            print("\n⏸️  用户暂停，正在保存进度...")
            self._save_metadata()
            print("✅ 进度已保存")
            return

        self._save_metadata()
        print(f"\n🎉 全部爬取完成！共 {len(self.metadata)} 张")
        for style in self.styles:
            count = len([item for item in self.metadata if item['style'] == style])
            print(f"   {style}: {count} 张")


if __name__ == "__main__":
    crawler = SmartCrawler()
    crawler.crawl_until(target_per_style=10000)