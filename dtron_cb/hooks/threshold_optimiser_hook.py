from typing import Tuple
import gc
from time import sleep
import json

from matplotlib import pyplot as plt
import numpy as np
from detectron2.structures.masks import BitMasks
import cv2
import torch

from ..config import CfgNode
from .base import HookBase


class ThresholdOptimiserHook(HookBase):

    def __init__(self, config: CfgNode):
        self.enabled = config.DATASETS.TEST_FRACTION > 0.0
        self.px_thresholds = np.linspace(
            config.THRESH_OPT.PIXEL_THRESH_MIN,
            config.THRESH_OPT.PIXEL_THRESH_MAX,
            config.THRESH_OPT.PIXEL_THRESH_N
        )
        self.ov_thresholds = np.linspace(
            config.THRESH_OPT.OVERALL_THRESH_MIN,
            config.THRESH_OPT.OVERALL_THRESH_MAX,
            config.THRESH_OPT.OVERALL_THRESH_N
        )
        self.n_measurements = config.THRESH_OPT.PIXEL_THRESH_N * config.THRESH_OPT.OVERALL_THRESH_N
        self.output_dir = config.OUTPUT_DIR

    def after_train(self):
        if self.enabled:
            px_thresh, ov_thrsh = self.do_thresh_opt()

            # run evaluation/metric calculation on test set, using thresholds defined above
            # TODO

    def do_thresh_opt(self) -> Tuple[float, float]:
        precisions = np.zeros(self.n_measurements)
        recalls = np.zeros(self.n_measurements)
        i = 0
        for px in self.px_thresholds:
            for ov in self.ov_thresholds:
                precisions[i], recalls[i] = self.get_precision_recall(px, ov)
                print(px, ov, precisions[i], recalls[i])
                i += 1

        # choose best thresholds
        dist_to_1 = np.sqrt(np.square(precisions - recalls))
        dist_to_1[~np.isfinite(dist_to_1)] = np.inf

        best_i = np.argmin(dist_to_1)
        print('min_dist_to_1', dist_to_1[best_i])
        best_px_thresh = self.px_thresholds[int(best_i / len(self.ov_thresholds))]
        best_ov_thresh = self.ov_thresholds[best_i % len(self.px_thresholds)]

        # plot
        plt.figure()
        plt.title(f'Best: Pixel Thresh = {best_px_thresh:.2f}, Score Thresh = {best_ov_thresh:.2f}')
        plt.plot(precisions, recalls, 'o')
        plt.plot([precisions[best_i], 1], [recalls[best_i], 1], 'o--')
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.xlim(left=-0.1, right=1.1)
        plt.ylim(bottom=-0.1, top=1.1)
        plt.plot([1.0], [1.0], 'kx')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig_precision_recall_curve.pdf')
        plt.close()

        with open(f'{self.output_dir}/recommended_thresholds.json', 'w') as f:
            json.dump(dict(ov_thresh=best_ov_thresh, px_thresh=best_px_thresh), f)

        return best_px_thresh, best_ov_thresh

    def get_gt_mask(self, instances):
        combined = np.zeros(instances.image_size, dtype=bool)
        masks = BitMasks.from_polygon_masks(instances.gt_masks, *instances.image_size).tensor.detach().cpu()
        for mask in masks:
            mask_bool = mask.numpy() > 0.5
            combined |= mask_bool
        return combined

    def get_outp_mask(self, instances, px_thresh, ov_thresh):
        combined = np.zeros(instances.image_size, dtype=bool)
        for score, maskf_t in zip(instances.scores, instances.pred_prob_masks):
            if score < ov_thresh:
                continue
            maskf = maskf_t.detach().cpu().numpy()
            mask_bool = maskf > px_thresh
            combined |= mask_bool
        return combined

    @staticmethod
    def compare_masks(gt, outp):
        gt = gt.flatten()
        outp = outp.flatten()

        t = gt == outp
        f = gt != outp
        p = gt
        n = ~gt

        tp = np.sum(t & p)
        fp = np.sum(f & p)
        tn = np.sum(t & n)
        fn = np.sum(f & n)

        return tp, fp, tn, fn

    def get_precision_recall(self, px_thresh: float, ov_thresh: float) -> Tuple[float, float]:
        if self.trainer.train_loader is not None:
            del self.trainer.train_loader
            self.trainer.train_loader = None
        torch.cuda.empty_cache()
        gc.collect()
        dl = self.trainer.valid_loader  # .to('cpu')
        model = self.trainer.model  #.to('cpu')
        model = model.eval()
        tp_t = fp_t = tn_t = fn_t = 0

        for batch in dl:
            for outp, dp in zip(model(batch), batch):
                sleep(0.01)
                gt_instances = dp['instances']
                gt_mask = self.get_gt_mask(gt_instances)
                outp_instances = outp['instances']
                outp_mask = self.get_outp_mask(outp_instances, px_thresh, ov_thresh)
                del outp
                del dp
                torch.cuda.empty_cache()
                gc.collect()
                gt_h, gt_w = gt_mask.shape
                outp_mask = cv2.resize(outp_mask.astype('uint8'), (gt_w, gt_h), interpolation=cv2.INTER_NEAREST).astype(bool)

                tp, fp, tn, fn = self.compare_masks(gt_mask, outp_mask)
                tp_t += int(tp)
                fp_t += int(fp)
                tn_t += int(tn)
                fn_t += int(fn)
        precision = np.divide(tp_t, tp_t + fp_t)
        recall = np.divide(tp_t, tp_t + fn_t)

        return precision, recall
