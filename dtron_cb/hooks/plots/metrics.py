from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from detectron2.engine.hooks import HookBase

from ...utils.read_metrics_json import read_metrics_json
from ...utils.string_san import path_safe


def metrics_plot(metrics_json_path: str, outfn: str = None):
    data = read_metrics_json(metrics_json_path)
    metrics = defaultdict(list)
    for i, d in enumerate(data):
        ti = d['iteration']
        for k in d:
            if k != 'iteration':
                metrics[k].append([ti, d[k]])

    for name, mdata in metrics.items():
        t, m = np.transpose(mdata)
        stor = ~np.isnan(m)
        t, m = t[stor], m[stor]
        plt.figure()
        if len(t) == 1:
            plt.plot(t, m, 'o')
        elif len(t) == 0:
            plt.plot([-1, 1], [-1, 1], 'white')
            plt.plot([1, -1], [-1, 1], 'white')
            plt.text(0, 0, 'No data', ha='center', va='center')
        else:
            plt.plot(t, m)
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.tight_layout()
        if outfn is None:
            plt.show()
        else:
            plt.savefig(outfn.format(metric=path_safe(name)))
        plt.close()


class MetricsPlotHook(HookBase):

    def __init__(self, cfg):
        self.output_dir = cfg.OUTPUT_DIR

    def after_train(self):
        metrics_plot(f'{self.output_dir}/metrics.json', f'{self.output_dir}/metrics_plot_{{metric}}.pdf')
