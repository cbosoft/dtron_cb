from typing import List

import numpy as np
from matplotlib import pyplot as plt
import cv2

from detectron2.engine.hooks import HookBase
from detectron2.data import DatasetCatalog


def plot_dataset(dataset: List[dict], cols=5, rows=None, max_n=100, fn: str = None):
    if len(dataset) > max_n:
        dataset = np.random.choice(dataset, max_n)
    if rows is None:
        rows = len(dataset) // cols
    sz = 7
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(cols * sz, rows * sz))
    for ax in axes.flatten():
        plt.sca(ax)
        plt.axis('off')
    for ax, d in zip(axes.flatten(), dataset):
        plt.sca(ax)
        im = cv2.imread(d['file_name'])
        plt.imshow(im)
        for annot in d['annotations']:
            ctr = annot['segmentation'][0]
            ctr_x = [*ctr[0::2], ctr[0]]
            ctr_y = [*ctr[1::2], ctr[1]]
            plt.plot(ctr_x, ctr_y, lw=3)
    plt.tight_layout()
    if fn:
        plt.savefig(fn)
    else:
        plt.show()
    plt.close()


class DatasetPlotHook(HookBase):

    def __init__(self, cfg):
        self.datasets = []
        self.datasets.extend(cfg.DATASETS.TRAIN)
        self.datasets.extend(cfg.DATASETS.TEST)
        self.output_dir = cfg.OUTPUT_DIR

    def before_train(self):
        print('Plotting dataset{0}: {1}'.format('s' if len(self.datasets) > 1 else '', ','.join(self.datasets)))
        for d in self.datasets:
            ds = DatasetCatalog.get(d)
            plot_dataset(ds, fn=f'{self.output_dir}/dataset_{d}.pdf')
