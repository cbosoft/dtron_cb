import time
import datetime
import logging
import torch
from collections import defaultdict

import numpy as np
import cv2

from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm

from .base import HookBase
from ..utils.instances_to_mask import instances_to_mask


class LossEvalHook(HookBase):
    def __init__(self, cfg, model):
        self._model: torch.nn.Module = model
        self._period = cfg.TEST.EVAL_PERIOD
        self._data_loader = build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
            DatasetMapper(cfg, True)
        )

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        metrics = defaultdict(list)
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
            for k, v in self._get_metrics(inputs).items():
                metrics[k].append(v)
        mean_loss = np.mean(losses)
        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
        self.trainer.storage.put_scalars(validation_loss=mean_loss, **mean_metrics)
        comm.synchronize()

        return losses

    def _get_loss(self, data: dict) -> float:
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def _get_metrics(self, data: dict) -> dict:
        self._model.eval()
        with torch.no_grad():
            try:
                gt = instances_to_mask(data[0]['instances'])
                pred = instances_to_mask(self._model(data)[0]['instances'], score_thresh=self._model.ov_thresh)
            except ValueError:
                print('Failed to calculate metrics due to mask error')
                self._model.train()
                return {}
            pred = cv2.resize(pred.astype(np.uint8), gt.shape[::-1]).astype(bool)
            assert gt.shape == pred.shape, f'Ground truth and prediction need to be the same size: {gt.shape} != {pred.shape}'
            tp = np.sum(gt & pred)
            fp = np.sum((~gt) & pred)
            fn = np.sum(gt & (~pred))
            tn = np.sum((~gt) & (~pred))
            p = tp + fn
            n = tn + fp
            metrics = dict(
                px_accuracy=(tp + tn) / (p + n),
                px_precision=tp/(tp + fp),
                px_recall=tp/p,
                px_f1=tp/(tp + .5*(fp + fn)),
                px_prevalence=p/(p + n),
                px_true_positive_rate=tp/p,
                px_false_positive_rate=fp/n,
                px_false_negative_rate=fn/p,
                px_true_negative_rate=tn/n
            )
            p = metrics['px_precision']
            r = metrics['px_recall']
            metrics['px_f0.5'] = (1. + 0.5*0.5)*p*r/(.5*.5*p + r)
            metrics['px_f2'] = (1. + 2*2)*p*r/(2*2*p + r)
        self._model.train()
        return metrics

    def after_epoch(self):
        self._do_loss_eval()
