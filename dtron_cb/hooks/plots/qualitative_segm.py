from typing import List, Tuple, Optional

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import cv2
import torch

from detectron2.engine.hooks import HookBase
from detectron2.data import DatasetCatalog
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from ...utils.instances_to_mask import instances_to_mask
from ...config import CfgNode


Int4 = Tuple[int, int, int, int]


def plot_qualitative_segm(dataset: List[dict], model, rows=4, fn: str = None, w=3, px_thresh=0.5, ov_thresh=0.5, crop: Optional[Int4] = None):
    original_px_thresh = px_thresh
    model.px_thresh = -0.1
    if len(dataset) > rows:
        dataset = np.random.choice(dataset, rows)
    sz = 7
    titles = ['Original', 'Ground Truth', 'Prediction', 'GT Mask', 'Predicted Mask']
    fig, axes = plt.subplots(ncols=len(titles), nrows=len(dataset), figsize=(sz*len(titles), rows*sz), squeeze=False)
    for ax in axes.flatten():
        plt.sca(ax)
        plt.axis('off')
    for ax, ttl in zip(axes.flatten(), titles):
        plt.sca(ax)
        plt.title(ttl)
    for (orig_ax, gt_ax, pred_ax, gt_mask_ax, pred_mask_ax), d in zip(axes, dataset):
        plt.sca(orig_ax)
        im = cv2.imread(d['file_name'])
        if im is None:
            plt.text(0, 0, 'imread failed')
            continue
        if crop:
            im = T.CropTransform(*crop).apply_image(im)
        plt.imshow(im)

        plt.sca(gt_ax)
        plt.imshow(im)
        for annot in d['annotations']:
            ctr = annot['segmentation'][0]
            ctr_x = [*ctr[0::2], ctr[0]]
            ctr_y = [*ctr[1::2], ctr[1]]
            if crop is not None:
                l, t, w, h = crop
                ctr_x = [max(min(cx, l+w), l) for cx in ctr_x]
                ctr_y = [max(min(cy, t+h), t) for cy in ctr_y]
            plt.plot(ctr_x, ctr_y, lw=3)

        plt.sca(pred_ax)
        inp = torch.tensor(im)
        inp = torch.permute(inp, (2, 0, 1))
        inst = model([dict(image=inp)])[0]['instances']
        height, width = im.shape[:2]
        composite = cv2.cvtColor(im, cv2.COLOR_RGB2BGRA).astype(float)/255
        composite[:, :, -1] = 1.0

        for i, (mask_t, score) in enumerate(zip(inst.pred_masks, inst.scores)):
            score = float(score.cpu())
            if score < ov_thresh:
                continue
            mask: np.ndarray = mask_t.cpu().numpy()

            submask = plt.cm.viridis(mask*255)
            submask[:, :, 3] = np.where(mask < 0.1, 0.0, 0.2)
            alpha_s = submask[:, :, 3]
            alpha_c = 1.0 - alpha_s
            for c in range(3):
                composite[:, :, c] = submask[:, :, c]*alpha_s + composite[:, :, c]*alpha_c
        plt.imshow(composite)
        smap = ScalarMappable(cmap='viridis')
        plt.colorbar(smap).ax.set_ylabel('Score [0,1]')

        for i, (bbox, score) in enumerate(zip(inst.pred_boxes, inst.scores)):
            score = float(score.cpu())
            if score < ov_thresh: continue
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
            x = [x1, x2, x2, x1, x1]
            y = [y1, y1, y2, y2, y1]
            plt.plot(x, y, color=plt.cm.viridis(score))

        # TODO GT_MASK

        plt.sca(pred_mask_ax)
        mask = instances_to_mask(inst, score_thresh=ov_thresh, mask_thresh=px_thresh)
        plt.imshow(mask, cmap='gray')

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()

    model.px_thresh = original_px_thresh


class QualitativeSegmHook(HookBase):

    def __init__(self, cfg: CfgNode, model):
        self.test = cfg.DATASETS.TEST
        self.output_dir = cfg.OUTPUT_DIR
        # self.init_cfg = cfg.clone()
        # self.final_cfg = cfg.clone()
        # self.init_cfg.defrost()
        # self.final_cfg.defrost()
        # self.final_cfg.MODEL.WEIGHTS = f'{self.output_dir}/model-final.pth'
        self.model = model
        self.crop = cfg.DATA.CROP
        self.ov_thresh = cfg.INFERENCE.OVERALL_THRESH
        self.px_thresh = cfg.INFERENCE.PIXEL_THRESH

    def before_train(self):
        self.model.eval()
        with torch.no_grad():
            for testn in self.test:
                dataset = DatasetCatalog.get(testn)
                plot_qualitative_segm(dataset, self.model, fn=f'{self.output_dir}/qualitative_segm_before_{testn}.pdf',
                                      crop=self.crop, px_thresh=self.px_thresh, ov_thresh=self.ov_thresh)
        self.model.train()

    def after_train(self):
        self.model.eval()
        with torch.no_grad():
            for testn in self.test:
                dataset = DatasetCatalog.get(testn)
                plot_qualitative_segm(dataset, self.model, fn=f'{self.output_dir}/qualitative_segm_after_{testn}.pdf',
                                      crop=self.crop, px_thresh=self.px_thresh, ov_thresh=self.ov_thresh)
        self.model.train()
