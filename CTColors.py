from torch.utils.data import DataLoader
from segmentDsets import Luna2dSegmentationDataset,  getCt
from segmentModel import UNetWrapper
import torch
import scipy.ndimage.morphology as morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def color_segmentation_by_regions(ct_slice, binary_mask, min_region_size=50, alpha=0.5):
    """
    根据连通区域给分割结果着色

    参数:
        ct_slice: 2D numpy数组，CT切片数据
        binary_mask: 2D numpy数组，二值分割掩码
        min_region_size: int，最小保留区域大小（像素数）
        alpha: float，分割掩码的透明度

    返回:
        着色后的RGB图像
    """
    # 标记连通区域
    labeled_regions, num_regions = ndimage.label(binary_mask)

    # 计算每个区域的大小
    region_sizes = ndimage.sum(np.ones_like(binary_mask), labeled_regions,
                               index=np.arange(1, num_regions + 1))

    # 过滤小区域
    valid_regions = np.zeros_like(labeled_regions)
    for i in range(1, num_regions + 1):
        if region_sizes[i - 1] >= min_region_size:
            valid_regions[labeled_regions == i] = i

    # 为每个区域分配随机颜色
    np.random.seed(42)  # 固定随机种子，确保结果可重现
    colors = np.random.rand(num_regions + 1, 3)  # +1 是为了包含背景0
    colors[0] = [0, 0, 0]  # 背景为黑色

    # 创建彩色图像
    color_image = np.zeros((ct_slice.shape[0], ct_slice.shape[1], 3), dtype=np.float32)

    # 为每个有效区域着色
    for i in np.unique(valid_regions):
        if i == 0:  # 跳过背景
            continue
        mask = valid_regions == i
        color_image[mask] = colors[int(i)]

    # 合并CT图像和彩色分割结果
    ct_normalized = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    ct_rgb = np.stack([ct_normalized] * 3, axis=-1)

    # 按alpha混合
    blended = ct_rgb * (1 - alpha) + color_image * alpha

    return blended


# 使用示例
def visualize_current_slice(ct, slice_index, output_a, threshold=0.5):
    """
    可视化指定索引的CT切片及其分割结果

    参数:
        ct: CT对象，包含CT数据
        slice_index: int，切片索引
        output_a: numpy数组，分割模型输出
        threshold: float，二值化阈值
    """
    # 获取CT切片
    ct_slice = ct.hu_a[slice_index]

    # 获取分割掩码（二值化）
    segmentation_mask = output_a[slice_index] > threshold

    # 高级可视化：按连通区域着色
    colored_image = color_segmentation_by_regions(ct_slice, segmentation_mask)

    plt.figure(figsize=(12, 10))
    plt.imshow(colored_image)
    plt.title(f"CT Slice {slice_index} with Colored Regions")
    plt.axis('off')
    plt.show()

def initModels():
    seg_dict = torch.load('data-unversioned/seg/models/seg/seg_2025-07-06_13.41.16_augment.best.state')
    seg_model = UNetWrapper(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode='upconv',
    )
    seg_model.load_state_dict(seg_dict['model_state'])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    seg_model = seg_model.to(device)
    seg_model.eval()
    return seg_model
def initDl():
    train_ds = Luna2dSegmentationDataset(
        val_stride=0,
        isValSet_bool=False,
        contextSlices_count=3,
        series_uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
    )
    batch_size = 8
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers= 1,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl

def segmentCt(ct, seg_model, device):
    with torch.no_grad():
        output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
        seg_dl = initDl()
        for input_t, _, _, slice_ndx_list in seg_dl:
            input_g = input_t.to(device)
            prediction_g = seg_model(input_g)

            for i, slice_ndx in enumerate(slice_ndx_list):
                output_a[slice_ndx] = prediction_g[i].cpu().numpy()
                visualize_current_slice(ct, slice_ndx, output_a)
        mask_a = output_a > 0.5
        mask_a = morphology.binary_erosion(mask_a, iterations=1)
    return mask_a

if __name__ == '__main__':

    ct = getCt('1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860')
    model = initModels()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    segmentCt(ct, model, device)


