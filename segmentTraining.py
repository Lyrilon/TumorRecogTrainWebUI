import argparse
import datetime
import hashlib
import os
import shutil
import sys
import time
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from util.util import enumerateWithEstimate
from segmentDsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from segmentModel import UNetWrapper, SegmentationAugmentation
from util.logconf import logging



# 与 app.py 保持一致的前端接口地址
BASE_URL = "http://127.0.0.1:5000"

# 日志配置
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Metrics 索引常量
METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_TN_NDX = 10
METRICS_SIZE = 11

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', default=16, type=int,
                            help='设定每个训练批次的数据加载量')
        parser.add_argument('--num-workers', default=4, type=int,
                            help='设定用于后台加载数据的工作进程数')
        parser.add_argument('--epochs', default=10, type=int,
                            help='设定训练轮数')
        parser.add_argument('--balanced', action='store_true', default=False,
                            help='是否对样本进行平衡采样')
        # 数据增强参数，支持连字符和下划线形式
        parser.add_argument('--augmented', '--augmented', dest='augmented',
                            action='store_true', default=False,
                            help='整体开启数据增强')
        parser.add_argument('--augment-flip', '--augment_flip', dest='augment_flip',
                            action='store_true', default=False,
                            help='开启翻转增强')
        parser.add_argument('--augment-offset', '--augment_offset', dest='augment_offset',
                            action='store_true', default=False,
                            help='开启水平偏移增强')
        parser.add_argument('--augment-scale', '--augment_scale', dest='augment_scale',
                            action='store_true', default=False,
                            help='开启缩放增强')
        parser.add_argument('--augment-rotate', '--augment_rotate', dest='augment_rotate',
                            action='store_true', default=True,
                            help='开启旋转增强')
        parser.add_argument('--augment-noise', '--augment_noise', dest='augment_noise',
                            action='store_true', default=True,
                            help='开启噪声增强')
        parser.add_argument('--tb-prefix', default='seg',
                            help='tensorboard 日志前缀')
        parser.add_argument('comment', nargs='?', default='none',
                            help='tensorboard 日志后缀')
        self.cli_args = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        # 构建数据增强配置
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # 检查 CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # 初始化模型与优化器
        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = UNetWrapper(
            in_channels=7, n_classes=1, depth=3, wf=4,
            padding=True, batch_norm=True, up_mode='upconv'
        )
        aug_model = SegmentationAugmentation(**self.augmentation_dict)
        if self.use_cuda:
            log.info(f'Using CUDA; {torch.cuda.device_count()} devices.')
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                aug_model = nn.DataParallel(aug_model)
            model = model.to(self.device)
            aug_model = aug_model.to(self.device)
        return model, aug_model

    def initOptimizer(self):
        return SGD(self.segmentation_model.parameters(), lr=1e-4, momentum=0.85)

    def initTrainDl(self):
        ds = TrainingLuna2dSegmentationDataset(
            val_stride=10, isValSet_bool=False, contextSlices_count=3)
        bs = self.cli_args.batch_size
        if self.use_cuda:
            bs *= torch.cuda.device_count()
        return DataLoader(ds, batch_size=bs,
                          num_workers=self.cli_args.num_workers,
                          pin_memory=self.use_cuda)

    def initValDl(self):
        ds = Luna2dSegmentationDataset(
            val_stride=10, isValSet_bool=True, contextSlices_count=3)
        bs = self.cli_args.batch_size
        if self.use_cuda:
            bs *= torch.cuda.device_count()
        return DataLoader(ds, batch_size=bs,
                          num_workers=self.cli_args.num_workers,
                          pin_memory=self.use_cuda)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir + '_trn_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir + '_val_' + self.cli_args.comment)

    def main(self):
        log.info(f'Starting training: {self.cli_args}')
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        best_score = 0.0
        for epoch in range(1, self.cli_args.epochs + 1):
            log.info(f'Epoch {epoch}/{self.cli_args.epochs}')
            trn_metrics = self.doTraining(epoch, train_dl)
            self.logMetrics(epoch, 'train', trn_metrics)
            val_metrics = self.doValidation(epoch, val_dl)
            score = self.logMetrics(epoch, 'val', val_metrics)
            self.saveModel('seg', epoch, score > best_score)
            if score > best_score:
                best_score = score
        try:
            requests.post(f'{BASE_URL}/training_done', timeout=1)
            log.info('Notified front-end of completion')
        except:
            log.warning('Failed to notify front-end')

    def doTraining(self, epoch, dl):
        metrics_g = torch.zeros(METRICS_SIZE, len(dl.dataset), device=self.device)
        self.segmentation_model.train()
        dl.dataset.shuffleSamples()
        total_batches = len(dl)
        for idx, batch in enumerateWithEstimate(dl, f'E{epoch} Training', start_ndx=dl.num_workers):
            # 暂停控制
            while True:
                try:  # 轮询前端暂停状态
                    resp = requests.get(f'{BASE_URL}/get_pause_state', timeout=1)
                    if resp.json().get('paused', False):
                        time.sleep(1)
                        continue
                except:
                    time.sleep(1)
                    continue
                break
            self.optimizer.zero_grad()
            loss = self.computeBatchLoss(idx, batch, dl.batch_size, metrics_g)
            loss.backward()
            self.optimizer.step()
            # 上报进度
            progress = {
                'epoch': epoch,
                'total_epochs': self.cli_args.epochs,
                'batch_iter': idx + 1,
                'total_batches': total_batches,
                'loss': round(loss.item(), 4),
                'accuracy': None,
            }
            try:
                requests.post(f'{BASE_URL}/update_progress', json=progress, timeout=1)
            except:
                pass
        self.totalTrainingSamples_count += metrics_g.size(1)
        return metrics_g.to('cpu')

    def doValidation(self, epoch, dl):
        with torch.no_grad():
            metrics_g = torch.zeros(METRICS_SIZE, len(dl.dataset), device=self.device)
            self.segmentation_model.eval()
            for idx, batch in enumerateWithEstimate(dl, f'E{epoch} Validation', start_ndx=dl.num_workers):
                self.computeBatchLoss(idx, batch, dl.batch_size, metrics_g)
        return metrics_g.to('cpu')

    def computeBatchLoss(self, idx, batch, bs, metrics_g, thresh=0.5):
        inp, label, _, _ = batch
        inp_g = inp.to(self.device, non_blocking=True)
        # 强制将标签转为 float，避免 bool 运算错误
        label_g = label.to(self.device, non_blocking=True).float()

        if self.augmentation_model and self.segmentation_model.training:
            inp_g, label_g = self.augmentation_model(inp_g, label_g)
            # SegmentationAugmentation returns a boolean mask, convert back to float tensor
            label_g = label_g.to(self.device, non_blocking=True).float()

        pred = self.segmentation_model(inp_g)
        dice = self.diceLoss(pred, label_g)
        fn_loss = self.diceLoss(pred * label_g, label_g)

        start = idx * bs
        end = start + inp.size(0)
        with torch.no_grad():
            pb = (pred[:, 0:1] > thresh).float()
            tp = (pb * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - pb) * label_g).sum(dim=[1, 2, 3])
            fp = (pb * (1 - label_g)).sum(dim=[1, 2, 3])
            tn = ((1 - pb) * (1 - label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start:end] = dice
            metrics_g[METRICS_TP_NDX, start:end]   = tp
            metrics_g[METRICS_FN_NDX, start:end]   = fn
            metrics_g[METRICS_FP_NDX, start:end]   = fp
            metrics_g[METRICS_TN_NDX, start:end]   = tn

        return dice.mean() + fn_loss.mean() * 8

    def diceLoss(self, pred, label, eps=1):
        l_sum = label.sum(dim=[1, 2, 3])
        p_sum = pred.sum(dim=[1, 2, 3])
        corr = (pred * label).sum(dim=[1, 2, 3])
        return 1 - (2 * corr + eps) / (p_sum + l_sum + eps)

    def logMetrics(self, epoch, mode, metrics_t):
        arr = metrics_t.detach().cpu().numpy()
        sums = arr.sum(axis=1)
        precision = sums[METRICS_TP_NDX] / ((sums[METRICS_TP_NDX] + sums[METRICS_FP_NDX]) or 1)
        recall    = sums[METRICS_TP_NDX] / ((sums[METRICS_TP_NDX] + sums[METRICS_FN_NDX]) or 1)
        metrics_dict = {
            'loss/all': arr[METRICS_LOSS_NDX].mean(),
            'pr/precision': precision,
            'pr/recall': recall,
            'pr/f1_score': 2 * precision * recall / ((precision + recall) or 1),
        }
        self.initTensorboardWriters()
        writer = getattr(self, mode + '_writer')
        for key, val in metrics_dict.items():
            writer.add_scalar('seg_' + key, val, self.totalTrainingSamples_count)
        writer.flush()
        return recall

    def logImages(self, epoch, mode, dl):
        self.segmentation_model.eval()
        writer = getattr(self, mode + '_writer')
        for i, uid in enumerate(sorted(dl.dataset.series_list)[:12]):
            ct = getCt(uid)
            for j in range(6):
                slice_idx = j * (ct.hu_a.shape[0] - 1) // 5
                ct_t, label_t, _, _ = dl.dataset.getitem_fullSlice(uid, slice_idx)
                inp = ct_t.unsqueeze(0).to(self.device)
                lab = label_t.unsqueeze(0).to(self.device)
                pred = self.segmentation_model(inp)[0]
                pred_mask = pred.cpu().detach().numpy()[0] > 0.5
                label_mask = lab.cpu().numpy()[0] > 0.5
                slice_img = ct_t[dl.dataset.contextSlices_count]
                image = np.zeros((512, 512, 3), dtype=np.float32)
                image[:, :, 0] = slice_img
                # 红色：FP 和 TP； 绿色：FN
                image[:, :, 0] += pred_mask & (~label_mask)
                image[:, :, 0] += pred_mask & label_mask
                image[:, :, 1] += (~pred_mask) & label_mask
                image = np.clip(image * 0.5, 0, 1)
                writer.add_image(f'{mode}/{i}_{j}', image, self.totalTrainingSamples_count, dataformats='HWC')
        writer.flush()

    def saveModel(self, prefix, epoch, isBest):
        base = os.path.join('data-unversioned', 'seg-checkPoint', 'models', self.cli_args.tb_prefix)
        os.makedirs(base, exist_ok=True)
        filename = f"{prefix}_{self.time_str}_{self.cli_args.comment}.{self.totalTrainingSamples_count}.state"
        path = os.path.join(base, filename)
        model = self.segmentation_model.module if isinstance(self.segmentation_model, nn.DataParallel) else self.segmentation_model
        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, path)
        log.info(f'Saved model to {path}')
        if isBest:
            best_path = path.replace('.state', '.best.state')
            shutil.copyfile(path, best_path)
            log.info(f'Saved best model to {best_path}')
        with open(path, 'rb') as f:
            log.info('SHA1: ' + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':

    
    SegmentationTrainingApp().main()
