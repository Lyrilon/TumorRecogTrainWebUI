import argparse
import random
import base64
import requests
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from segmentModel import UNetWrapper
from segmentDsets import Luna2dSegmentationDataset, getCt
from util.logconf import logging

# 与 app.py 保持一致的前端接口地址
BASE_URL = "http://127.0.0.1:5000"

# 日志配置
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def array_to_data_url(arr: np.ndarray) -> str:
    """
    将单通道 numpy 数组转为 base64 编码的 PNG Data URL
    用于原始输入图片和标签，保持灰度显示
    """
    # 归一化到 0-1 范围，防止数据溢出或丢失细节
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def overlay_mask_on_image(
    image_arr: np.ndarray, mask_arr: np.ndarray, color=(255, 0, 0), alpha=0.3
) -> str:
    """
    将二值掩码以指定颜色和透明度叠加到灰度图像上，并返回 base64 编码的 PNG Data URL
    :param image_arr: 原始灰度图像的 numpy 数组 (H, W)
    :param mask_arr: 模型预测的二值掩码的 numpy 数组 (H, W)，值应为 0 或 1
    :param color: 叠加掩码的颜色 (R, G, B)，默认为红色
    :param alpha: 叠加掩码的透明度，范围 0.0 到 1.0
    :return: 叠加后的图像的 base64 编码的 Data URL
    """
    # 归一化输入图像到 0-255 范围
    norm_image_arr = (
        (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min() + 1e-6) * 255
    )
    norm_image_arr = norm_image_arr.astype(np.uint8)

    # 将灰度图像转换为 RGB 格式
    rgb_image = Image.fromarray(norm_image_arr).convert("RGB")
    rgb_image_np = np.array(rgb_image)  # 转换为 numpy 数组 (H, W, 3)

    # 将掩码转换为 RGB 颜色，并应用透明度
    # 创建一个与图像大小相同的全透明层
    overlay = np.zeros_like(rgb_image_np, dtype=np.uint8)

    # 只有 mask_arr 中为 1 的地方才填充颜色
    # 注意：mask_arr 应该是 (H, W) 的二值数组
    mask_indices = mask_arr == 1

    # 将指定颜色应用到掩码区域
    overlay[mask_indices, 0] = color[0]  # R
    overlay[mask_indices, 1] = color[1]  # G
    overlay[mask_indices, 2] = color[2]  # B

    # 将叠加层转换为 PIL Image，并设置模式为 RGBA 以支持透明度
    overlay_img = Image.fromarray(overlay, mode="RGB")

    # 创建一个 Alpha 通道
    alpha_mask = np.zeros_like(mask_arr, dtype=np.uint8)
    alpha_mask[mask_indices] = int(alpha * 255)  # 设置透明度值
    alpha_img = Image.fromarray(alpha_mask, mode="L")  # L for grayscale

    overlay_img.putalpha(alpha_img)  # 将 alpha 通道添加到 overlay_img

    # 将原始 RGB 图像转换为 PIL Image 以进行叠加操作
    base_img = Image.fromarray(rgb_image_np, mode="RGB")

    # 使用 PIL 的 alpha_composite 进行叠加
    # 为了使用 alpha_composite，两张图片都必须有 alpha 通道
    # base_img 转换为 RGBA，alpha 通道为完全不透明
    base_img = base_img.convert("RGBA")

    # overlay_img 已经有 alpha 通道

    # 叠加图像
    # Image.alpha_composite 要求两个图片都是 RGBA 模式
    combined_img = Image.alpha_composite(base_img, overlay_img)

    # 将叠加后的图片保存到 BytesIO 并编码为 base64
    buf = BytesIO()
    combined_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_inference(model_path: str):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNetWrapper(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode="upconv",
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info(f"Loaded model from {model_path}")

    # 获取验证集所有 series
    ds = Luna2dSegmentationDataset(
        val_stride=10, isValSet_bool=True, contextSlices_count=3
    )
    series_list = sorted(ds.series_list)
    if not series_list:
        log.error("No validation series available.")
        return

    # 随机选一个 series 和 slice
    series_uid = random.choice(series_list)
    ct = getCt(series_uid)
    n_slices = ct.hu_a.shape[0]
    slice_idx = random.randrange(n_slices)

    # 获取数据
    # input_t 的形状是 (C, H, W)，其中 C 是 contextSlices_count * 2 + 1
    # 我们通常展示中间的那一层 (contextSlices_count) 作为输入图像
    input_t, label_t, _, _ = ds.getitem_fullSlice(series_uid, slice_idx)
    inp_batch = input_t.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        pred = (
            model(inp_batch)[0, 0].cpu().numpy()
        )  # [0,0] 表示取第一个样本的第一个通道
    pred_mask = (pred > 0.5).astype(np.float32)  # 二值化预测结果

    # 转 Data URL
    img_np = input_t[ds.contextSlices_count].numpy()  # 获取中间层的图像作为显示输入
    gt_np = label_t[0].numpy().astype(np.float32)  # 真实标签，通常也只有一个通道

    input_url = array_to_data_url(img_np)
    label_url = array_to_data_url(gt_np)

    # *** 关键改动：使用 overlay_mask_on_image 生成叠加图像的 URL ***
    # 将原始输入图像和预测的二值化掩码传给新函数
    output_url = overlay_mask_on_image(img_np, pred_mask, color=(255, 0, 0), alpha=0.4)

    payload = {
        "series_uid": series_uid,
        "slice_idx": slice_idx,
        "input_url": input_url,
        "label_url": label_url,
        "output_url": output_url,  # 现在 output_url 是叠加后的图片
    }
    try:
        requests.post(f"{BASE_URL}/update_inference_image", json=payload, timeout=2)
        log.info(f"Sent inference for series {series_uid}, slice {slice_idx}")
    except Exception as e:
        log.error(f"Failed to send inference image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation inference script")
    parser.add_argument("--eval", required=True, help="模型 checkpoint 路径")
    args = parser.parse_args()
    run_inference(args.eval)
