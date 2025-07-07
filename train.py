import argparse
import time
import requests
import random
import math

BASE_URL = "http://127.0.0.1:5000"


def simulate_training(args):
    """
    模拟模型训练过程。
    """
    total_batches = 100
    current_loss = 1.0
    current_acc = 0.6
    print("Training started with args:", args)

    for epoch in range(1, args.epochs + 1):
        for batch_iter in range(1, total_batches + 1):
            while True:
                try:
                    response = requests.get(f"{BASE_URL}/get_pause_state", timeout=1)
                    if response.json().get("paused", False):
                        print("Training paused...")
                        time.sleep(2)
                    else:
                        break
                except requests.exceptions.RequestException as e:
                    print(f"Could not connect to server to check pause state: {e}")
                    time.sleep(2)

            time.sleep(0.1)
            current_loss -= random.uniform(0.001, 0.005) * (1 / (epoch + 1))
            current_loss = max(0.01, current_loss)
            current_acc += random.uniform(0.001, 0.003) * (1 / (epoch + 1))
            current_acc = min(0.99, current_acc)

            progress_data = {
                "epoch": epoch,
                "total_epochs": args.epochs,
                "batch_iter": batch_iter,
                "total_batches": total_batches,
                "loss": round(current_loss, 4),
                "accuracy": round(current_acc, 4),
            }

            try:
                requests.post(
                    f"{BASE_URL}/update_progress", json=progress_data, timeout=1
                )
            except requests.exceptions.RequestException as e:
                print(f"Failed to send progress update: {e}")

    try:
        requests.post(f"{BASE_URL}/training_done", timeout=1)
        print("Training finished. Sent completion signal.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send completion signal: {e}")


def simulate_inference(args):
    """
    模拟模型推理过程。
    """
    print(f"Inference started for model: {args.eval}")

    # 模拟加载数据和模型需要一些时间
    time.sleep(2)

    # 准备用于前端显示的图片URL（使用占位图服务）
    # 1. 输入的CT图：灰色背景
    input_url = "https://placehold.co/400x400/333333/FFFFFF?text=Input%5CnCT%20Scan"
    # 2. 真实标签图：黑色背景，白色标注区域
    label_url = (
        "https://placehold.co/400x400/000000/FFFFFF?text=Ground%20Truth%5Cn(Label)"
    )
    # 3. 模型输出图：灰色背景，用红色区域模拟U-Net的分割结果
    output_url = "https://placehold.co/400x400/333333/FF0000?text=Model%20Output%5Cn(Red%20Nodule)"

    image_data = {
        "input_url": input_url,
        "label_url": label_url,
        "output_url": output_url,
    }

    # 将图片URL发送回Flask服务器
    try:
        requests.post(f"{BASE_URL}/update_inference_image", json=image_data, timeout=2)
        print("Successfully sent inference image URLs to server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send inference image URLs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulated Model Training and Inference Script"
    )
    # 训练参数
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--balanced", action="store_true", default=False)
    parser.add_argument("--augmented", action="store_true", default=False)
    parser.add_argument("--augment-flip", action="store_true", default=False)
    parser.add_argument("--augment-offset", action="store_true", default=False)
    parser.add_argument("--augment-scale", action="store_true", default=False)
    parser.add_argument("--augment-rotate", action="store_true", default=False)
    parser.add_argument("--augment-noise", action="store_true", default=False)

    # 新增的推理参数
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Path to the model for evaluation/inference.",
    )

    args = parser.parse_args()

    # 根据是否存在 --eval 参数来决定执行训练还是推理
    if args.eval:
        simulate_inference(args)
    else:
        simulate_training(args)
