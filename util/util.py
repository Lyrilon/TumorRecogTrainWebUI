import collections
import datetime
import time
import numpy as np
import math
from util.logconf import logging  # 从自定义的logconf模块中导入logging设置

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)
log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，表示输出所有日志信息（用于调试）

# 定义三元组，表示体素坐标：index（切片层）、row（行）、col（列）
VoxelCoordTuple = collections.namedtuple('VoxelCoordTuple', ['index', 'row', 'col'])

# 定义三元组，表示患者实际空间坐标：x、y、z（三维坐标）
PatientCoordTuple = collections.namedtuple('PatientCoordTuple', ['x', 'y', 'z'])


# 体素坐标转病人坐标（图像坐标 -> 实际空间坐标）
def voxelCoord2patientCoord(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    # coord_irc 是 (index, row, col)，这里需要转换为 (z, y, x) 顺序
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)

    # 应用方向矩阵和体素大小，计算实际的空间坐标
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a

    return PatientCoordTuple(*coords_xyz)

# 病人坐标转体素坐标（实际空间坐标 -> 图像坐标）
def patientCoord2voxelCoord(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)

    # 反向应用方向矩阵，并除以体素大小，得到体素坐标（z, y, x顺序）
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)  # 四舍五入取整，转换为离散坐标

    # 将 (z, y, x) 转换为 (index, row, col)
    return VoxelCoordTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

# 动态导入模块或对象
def importstr(module_str, from_=None):
    """
    动态导入模块或模块中的对象，例如：
    >>> importstr('os')         # 返回os模块
    >>> importstr('math', 'fabs')  # 返回math.fabs函数
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))

    return module


# 枚举并估计剩余时间，同时记录日志
def enumerateWithEstimate(
        iterable,        # 可迭代对象
        desc_str,        # 描述字符串，例如 "Stuffing cache"
        start_ndx=0,     # 从哪个下标开始计算耗时
        print_ndx=4,     # 第几个元素开始第一次打印
        backoff=2,       # 每次打印后，间隔乘以这个因子
        iter_len=None    # 迭代器长度（若未指定将自动推断）
):
    # 1. 尝试推断长度
    if iter_len is None:
        try:
            iter_len = len(iterable)
        except (TypeError, AttributeError):
            iter_len = None

    # 2. 初始化
    start_time = time.time()
    next_print = print_ndx

    # 3. 依次迭代
    for idx, item in enumerate(iterable):
        # 3.1 打印进度
        if idx >= next_print:
            elapsed = time.time() - start_time
            count = idx - start_ndx + 1
            per_item = elapsed / count if count > 0 else 0
            if iter_len:
                remaining = per_item * (iter_len - idx - 1)
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
                print(f"{desc_str}: {idx+1}/{iter_len} "
                      f"elapsed {elapsed:.1f}s, ETA {eta}")
            else:
                print(f"{desc_str}: {idx+1} items, elapsed {elapsed:.1f}s")

            # 更新下次打印的阈值
            next_print = math.ceil(next_print * backoff)

        yield idx, item