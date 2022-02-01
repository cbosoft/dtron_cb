import torch.nn as nn

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

from .hooks import (
    LossEvalHook,
    DatasetPlotHook,
    QualitativeSegmHook,
    MetricsPlotHook,
    CopyCompleteHook,
    WriteMetaHook
)


class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        # self.model: nn.Module = nn.Module()
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.extend([
            LossEvalHook(self.cfg, self.model),
            DatasetPlotHook(self.cfg),
            QualitativeSegmHook(self.cfg, self.model),
            MetricsPlotHook(self.cfg),
            WriteMetaHook(self.cfg, self.model)
        ])

        # Must be last
        hooks.append(CopyCompleteHook(self.cfg))
        return hooks

    def build_writers(self):
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(None),
            JSONWriter(f'{self.cfg.OUTPUT_DIR}/metrics.json'),
        ]

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [eval(augsrc, dict(T=T)) for augsrc in cfg.DATA.AUGMENTATIONS]
        if cfg.DATA.CROP:
            augs.insert(0, T.CropTransform(*cfg.DATA.CROP))
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
