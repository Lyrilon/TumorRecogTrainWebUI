import argparse
import datetime
import hashlib
import os
import shutil
import sys
import time
import requests  # enable front-end communication

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import TumorDatasets
import TumorModel

from util.util import enumerateWithEstimate
from util.logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Web UI default endpoint
DEFAULT_SERVER = "http://127.0.0.1:5000"

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4

class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', default=24, type=int,
                            help='Batch size per device')
        parser.add_argument('--num-workers', default=4, type=int,
                            help='Number of DataLoader workers')
        parser.add_argument('--epochs', default=5, type=int,
                            help='Total training epochs')
        parser.add_argument('--dataset', default='MalignantLunaDataset',
                            help='Dataset class name in TumorDatasets')
        parser.add_argument('--model', default='LunaModel',
                            help='Model class name in TumorModel')
        parser.add_argument('--finetune', default='',
                            help='Checkpoint path to fine-tune from, empty to disable')
        parser.add_argument('--finetune-depth', default=2, type=int,
                            help='Number of trailing layers to fine-tune')
        parser.add_argument('--tb-prefix', default='tumor_cls',
                            help='TensorBoard log directory prefix')
        parser.add_argument('--server-url', default=DEFAULT_SERVER,
                            help='Base URL for front-end server')
        parser.add_argument('comment', nargs='?', default='run',
                            help='Experiment comment suffix')

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.server = self.cli_args.server_url.rstrip('/')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        # Data augmentation settings
        self.augmentation_dict = dict(flip=True, offset=0.1,
                                      scale=0.2, rotate=True,
                                      noise=25.0)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model_cls = getattr(TumorModel, self.cli_args.model)
        model = model_cls()
        if self.cli_args.finetune:
            ckpt = torch.load(self.cli_args.finetune, map_location='cpu')
            blocks = [n for n, m in model.named_children() if any(p.requires_grad for p in m.parameters())]
            tune = blocks[-self.cli_args.finetune_depth:]
            log.info(f"Fine-tuning from {self.cli_args.finetune}, blocks={tune}")
            model.load_state_dict({k: v for k, v in ckpt['model_state'].items() if k.split('.')[0] not in tune}, strict=False)
            for name, param in model.named_parameters():
                if name.split('.')[0] not in tune:
                    param.requires_grad_(False)
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)
            log.info(f"Using CUDA devices: {torch.cuda.device_count()}")
        return model

    def initOptimizer(self):
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir + '-trn-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir + '-val-' + self.cli_args.comment)

    def initTrainDl(self):
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)
        train_ds = ds_cls(val_stride=10, isValSet_bool=False, ratio_int=1,
                          augmentation_dict=self.augmentation_dict)
        bs = self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1)
        return DataLoader(train_ds, batch_size=bs, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

    def initValDl(self):
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)
        val_ds = ds_cls(val_stride=10, isValSet_bool=True)
        bs = self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1)
        return DataLoader(val_ds, batch_size=bs, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

    def main(self):
        self.initTensorboardWriters()

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        best_score = 0.0
        for epoch in range(1, self.cli_args.epochs + 1):
            try:
                requests.post(f"{self.server}/training_started", json={'epoch': epoch}, timeout=1)
            except:
                pass

            trn_metrics = self.doTraining(epoch, train_dl)
            val_metrics = self.doValidation(epoch, val_dl)

            # Compute detailed epoch stats
            avg_trn_loss = trn_metrics[METRICS_LOSS_NDX].mean().item()
            avg_val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
            val_labels = val_metrics[METRICS_LABEL_NDX].cpu().numpy()
            val_preds = val_metrics[METRICS_PRED_NDX].cpu().numpy()
            val_acc = float((val_labels == val_preds).sum() / len(val_labels))
            from sklearn.metrics import roc_auc_score
            val_probs = val_metrics[METRICS_PRED_P_NDX].cpu().numpy()
            try:
                val_auc = float(roc_auc_score(val_labels, val_probs))
            except Exception:
                val_auc = 0.0

            self.trn_writer.add_scalar('loss/train', avg_trn_loss, epoch)
            self.val_writer.add_scalar('loss/val', avg_val_loss, epoch)
            self.val_writer.add_scalar('acc/val', val_acc, epoch)
            self.val_writer.add_scalar('auc/val', val_auc, epoch)

            try:
                summary = {'epoch': epoch,
                           'total_epochs': self.cli_args.epochs,
                           'train_loss': round(avg_trn_loss, 4),
                           'val_loss': round(avg_val_loss, 4),
                           'val_accuracy': round(val_acc, 4),
                           'val_auc': round(val_auc, 4)}
                requests.post(f"{self.server}/update_epoch", json=summary, timeout=1)
            except:
                pass

            current_score = val_auc
            is_best = current_score > best_score
            self.saveModel('seg', epoch, is_best)
            if is_best:
                best_score = current_score

        try:
            requests.post(f"{self.server}/training_done", timeout=1)
        except:
            pass

    def computeBatchLoss(self, batch_ndx, batch, batch_size, metrics, augment=True):
        inputs, labels, idxs, *_ = batch
        g_in = inputs.to(self.device)
        g_lbl = labels.to(self.device)
        if augment:
            g_in = TumorModel.augment3d(g_in)
        logits, probs = self.model(g_in)
        loss = nn.functional.cross_entropy(logits, g_lbl[:, 1], reduction='none')
        start = batch_ndx * batch_size
        end = start + labels.size(0)
        pred = torch.argmax(probs, dim=1)
        metrics[METRICS_LABEL_NDX, start:end] = idxs.to(self.device)
        metrics[METRICS_PRED_NDX, start:end] = pred
        metrics[METRICS_PRED_P_NDX, start:end] = probs[:, 1]
        metrics[METRICS_LOSS_NDX, start:end] = loss
        return loss.mean()

    def doTraining(self, epoch, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        metrics = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        total_batches = len(train_dl)
        for batch_ndx, batch in enumerateWithEstimate(train_dl, f"E{epoch} Training", start_ndx=train_dl.num_workers):
            while True:
                try:
                    r = requests.get(f"{self.server}/get_pause_state", timeout=1)
                    if r.json().get('paused', False):
                        log.info("Training paused...")
                        time.sleep(2)
                        continue
                    break
                except requests.exceptions.RequestException as e:
                    log.warning(f"Pause check failed: {e}")
                    time.sleep(2)

            self.optimizer.zero_grad()
            loss = self.computeBatchLoss(batch_ndx, batch, train_dl.batch_size, metrics, augment=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                inputs, labels, idxs, *_ = batch
                g_in = inputs.to(self.device)
                logits, probs = self.model(g_in)
                pred = torch.argmax(probs, dim=1).cpu()
                correct = (pred == idxs).sum().item()
                batch_acc = correct / labels.size(0)

            progress = {'epoch': epoch,
                        'total_epochs': self.cli_args.epochs,
                        'batch_iter': batch_ndx,
                        'total_batches': total_batches,
                        'loss': round(loss.item(), 4),
                        'accuracy': round(batch_acc, 4)}
            try:
                requests.post(f"{self.server}/update_progress", json=progress, timeout=1)
            except:
                pass

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return metrics.to('cpu')

    def doValidation(self, epoch, val_dl):
        self.model.eval()
        metrics = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
        with torch.no_grad():
            for batch_ndx, batch in enumerateWithEstimate(val_dl, f"E{epoch} Validation", start_ndx=val_dl.num_workers):
                self.computeBatchLoss(batch_ndx, batch, val_dl.batch_size, metrics, augment=False)
        return metrics.to('cpu')

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join('data-unversioned', 'tumor_checkPoint', 'models',
                                 self.cli_args.tb_prefix,
                                 f"{type_str}_{self.time_str}_{self.cli_args.comment}.{self.totalTrainingSamples_count}.state")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state = {'model_state': model_to_save.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'epoch': epoch_ndx,
                 'totalTrainingSamples_count': self.totalTrainingSamples_count}
        torch.save(state, file_path)
        log.info(f"Saved model checkpoint to {file_path}")
        if isBest:
            best_path = os.path.join('data-unversioned', 'tumor', 'models',
                                     self.cli_args.tb_prefix,
                                     f"{type_str}_{self.time_str}_{self.cli_args.comment}.best.state")
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            shutil.copyfile(file_path, best_path)
            log.info(f"Copied best checkpoint to {best_path}")
        with open(file_path, 'rb') as f:
            sha1 = hashlib.sha1(f.read()).hexdigest()
            log.info(f"SHA1: {sha1}")

if __name__ == '__main__':
    ClassificationTrainingApp().main()
