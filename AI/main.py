import sys
from train import train
from inference import generate_image
from crawler import SmartCrawler

def auto_mode():
    print("\n🚀 启动自动模式（爬虫 → 训练）")
    
    crawler = SmartCrawler()
    current_count = len(crawler.metadata)
    print(f"📂 当前图片数量：{current_count}")

    if current_count < 30000:
        print("🕷️ 开始爬取数据（目标30000张）...")
        crawler.crawl_until()
    else:
        print("✅ 图片数量已满足，跳过爬取")

    print("\n🧠 开始训练...")
    train()
    
    print("\n🎉 自动模式完成！")
    print("💡 想生成图片？请运行 python main.py 选 3，然后输入提示词")


def main():
    print("=" * 50)
    print("🤖 MyAI Generator")
    print("=" * 50)
    print("【0】全自动模式（爬虫→训练）")
    print("【1】仅爬取数据（3万张）")
    print("【2】仅训练模型")
    print("【3】生成图片（需先训练完）")
    print("【4】退出")
    print("=" * 50)

    choice = input("请选择 [0/1/2/3/4]: ")

    if choice == "0":
        auto_mode()
    elif choice == "1":
        crawler = SmartCrawler()
        crawler.crawl_until()
    elif choice == "2":
        train()
    elif choice == "3":
        generate_image()
    elif choice == "4":
        print("👋 再见！")
        sys.exit(0)
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()