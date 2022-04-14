from collections import defaultdict
from typing import Dict
import re

import numpy as np
from matplotlib import pyplot as plt
from detectron2.engine.hooks import HookBase

from ...utils.read_metrics_json import read_metrics_json
from ...utils.string_san import path_safe


if hasattr(re, 'Pattern'):
    RegexPattern = re.Pattern
else:
    RegexPattern = str


class LabelsByPattern:

    def __init__(self, lbls_and_patterns: Dict[str, str]):
        self.labels_and_patterns = {k: re.compile(v) for k, v in lbls_and_patterns.items()}

    def get(self, name, default_label='iteration'):
        for lbl, pattern in self.labels_and_patterns.items():
            if pattern.match(name):
                return lbl
        return default_label


def metrics_plot(metrics_json_path: str, outfn: str = None,
                 groups: Dict[str, RegexPattern] = None,
                 x_kind: LabelsByPattern = None):
    data = read_metrics_json(metrics_json_path)
    metrics = defaultdict(list)
    x_kind = x_kind if x_kind else LabelsByPattern({})
    for i, d in enumerate(data):
        for k in d:
            if k not in ('iteration', 'train_batch', 'test_batch'):
                xk = x_kind.get(k, 'iteration')
                xv = d[xk]
                metrics[k].append([xv, d[k]])

    groups = groups if groups else {}

    grouped_data = {k: list() for k in groups}

    for name, mdata in metrics.items():
        x_label = x_kind.get(name, 'iteration')
        t, m = np.transpose(mdata)
        stor = ~np.isnan(m)
        t, m = t[stor], m[stor]
        for group, pattern in groups.items():
            match = pattern.match(name)
            if match:
                grouped_data[group].append([t, m, match.group(1)])
        plt.figure()
        if len(t) == 1:
            plt.plot(t, m, 'o')
        elif len(t) == 0:
            plt.plot([-1, 1], [-1, 1], 'white')
            plt.plot([1, -1], [-1, 1], 'white')
            plt.text(0, 0, 'No data', ha='center', va='center')
        else:
            plt.plot(t, m)
        plt.xlabel(x_label)
        plt.ylabel(name)
        plt.tight_layout()
        if outfn is None:
            plt.show()
        else:
            plt.savefig(outfn.format(metric=path_safe(name)))
        plt.close()

    for group, data in grouped_data.items():
        plt.figure()
        for t, m, lbl in data:
            plt.plot(t, m, label=lbl)
        plt.legend()
        plt.tight_layout()
        if outfn is None:
            plt.show()
        else:
            plt.savefig(outfn.format(metric=path_safe(group)))
        plt.close()


class MetricsPlotHook(HookBase):

    def __init__(self, cfg, groups=None, xlbls=None):
        self.output_dir = cfg.OUTPUT_DIR
        self.groups: Dict[str, RegexPattern] = {k: re.compile(v) for k, v in groups.items()} if groups else {}
        self.xlbls = LabelsByPattern(xlbls if xlbls else {})

    def after_train(self):
        if self.trainer.state != 'failed':
            metrics_plot(
                f'{self.output_dir}/metrics.json',
                f'{self.output_dir}/metrics_plot_{{metric}}.pdf',
                groups=self.groups,
                x_kind=self.xlbls
            )
