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
    """
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def run_inference(model_path: str):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = UNetWrapper(
        in_channels=7, n_classes=1, depth=3, wf=4,
        padding=True, batch_norm=True, up_mode='upconv'
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info(f"Loaded model from {model_path}")

    # 获取验证集所有 series
    ds = Luna2dSegmentationDataset(val_stride=10, isValSet_bool=True, contextSlices_count=3)
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
    input_t, label_t, _, _ = ds.getitem_fullSlice(series_uid, slice_idx)
    inp_batch = input_t.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        pred = model(inp_batch)[0, 0].cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.float32)

    # 转 Data URL
    img_np = input_t[ds.contextSlices_count].numpy()
    gt_np = label_t[0].numpy().astype(np.float32)
    input_url  = array_to_data_url(img_np)
    label_url  = array_to_data_url(gt_np)
    output_url = array_to_data_url(pred_mask)

    payload = {
        'series_uid': series_uid,
        'slice_idx': slice_idx,
        'input_url': input_url,
        'label_url': label_url,
        'output_url': output_url
    }
    try:
        requests.post(f"{BASE_URL}/update_inference_image", json=payload, timeout=2)
        log.info(f"Sent inference for series {series_uid}, slice {slice_idx}")
    except Exception as e:
        log.error(f"Failed to send inference image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation inference script')
    parser.add_argument('--eval', required=True, help='模型 checkpoint 路径')
    args = parser.parse_args()
    run_inference(args.eval)
