import logging
import weakref

import torch
from torch.utils.data import Dataset as _Dataset, DataLoader
import torch.nn as nn

from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, get_event_storage, EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.engine.hooks import (
    IterationTimer,
    LRScheduler,
    PeriodicCheckpointer,
    PeriodicWriter
)

from ..hooks import (
    LossEvalHook,
    DatasetPlotHook,
    QualitativeSegmHook,
    MetricsPlotHook,
    CopyCompleteHook,
    WriteMetaHook,
    DeployModelHook,
    DisplayProgressHook
)

from .base import TrainerBase


class Dataset(_Dataset):

    def __init__(self, data, mapper):
        self.mapper = mapper
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.mapper(self.data[item])


class Trainer(TrainerBase):

    def __init__(self, config):
        self.config = config
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        self.model = build_model(config)
        self.optimiser = build_optimizer(config, self.model)
        train_loader = self.build_train_loader(config)
        test_loader = self.build_test_loader(config)
        super().__init__(config.SOLVER.N_EPOCHS, train_loader, test_loader)
        self.scheduler = build_lr_scheduler(config, self.optimiser)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            config.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        print(self.model.device)

        self.register_hooks(self.build_hooks())

    @property
    def optimizer(self):
        """American spelling provided for compatibility."""
        return self.optimiser

    def build_hooks(self):
        hooks = [
            IterationTimer(),
            LRScheduler(),
            LossEvalHook(self.config, self.model),
            # DatasetPlotHook(self.config),
            QualitativeSegmHook(self.config, self.model),
            MetricsPlotHook(self.config,
                            groups=dict(
                                test_loss=r'loss/test/(.*)',
                                train_loss=r'loss/train/(.*)'
                            ),
                            # xlbls={'epoch': r'.*loss.*'}
            ),
            WriteMetaHook(self.config, self.model),
            DeployModelHook(self.config, self.model),
            PeriodicCheckpointer(self.checkpointer, self.config.SOLVER.CHECKPOINT_PERIOD),
            PeriodicWriter(self.build_writers(), period=1),
            DisplayProgressHook(),
            # Must be last
            CopyCompleteHook(self.config)
        ]
        return hooks

    def build_writers(self):
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            # CommonMetricPrinter(None),
            JSONWriter(f'{self.config.OUTPUT_DIR}/metrics.json'),
        ]

    @classmethod
    def build_train_loader(cls, config):
        augs = [eval(augsrc, dict(T=T)) for augsrc in config.DATA.AUGMENTATIONS]
        if config.DATA.CROP:
            augs.insert(0, T.CropTransform(*config.DATA.CROP))
        mapper = DatasetMapper(config, is_train=True, augmentations=augs)
        dicts = []
        for dsname in config.DATASETS.TRAIN:
            dicts.extend(DatasetCatalog.get(dsname))
        ds = Dataset(dicts, mapper)
        return DataLoader(ds, batch_size=config.SOLVER.IMS_PER_BATCH, shuffle=True, collate_fn=lambda b: b)

    @classmethod
    def build_test_loader(cls, config):
        augs = []
        if config.DATA.CROP:
            augs.insert(0, T.CropTransform(*config.DATA.CROP))
        # is_train = True so that the model output is more detailed, not post-processed etc
        mapper = DatasetMapper(config, is_train=True, augmentations=augs)
        dicts = []
        for dsname in config.DATASETS.TEST:
            dicts.extend(DatasetCatalog.get(dsname))
        ds = Dataset(dicts, mapper)
        return DataLoader(ds, batch_size=config.SOLVER.IMS_PER_BATCH, shuffle=True, collate_fn=lambda b: b)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.config.MODEL.WEIGHTS, resume=resume)
        # if resume and self.checkpointer.has_checkpoint():
        #     # The checkpoint stores the training iteration that just finished, thus we start
        #     # at the next iteration
        #     self.start_iter = self.iter + 1

    def do_train_batch(self, batch) -> float:
        loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self.optimiser.zero_grad()
        losses.backward()
        self.optimiser.step()

        storage: EventStorage = get_event_storage()
        renamed_loss_dict = dict()
        for k, v in loss_dict.items():
            k = k.replace('loss_', '')
            k = f'loss/train/{k}'
            renamed_loss_dict[k] = v

        storage.put_scalars(**renamed_loss_dict, **self.time_data(), smoothing_hint=False)

        return float(losses.detach().cpu().item())

    def do_test_batch(self, batch) -> float:
        loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        storage: EventStorage = get_event_storage()
        renamed_loss_dict = dict()
        for k, v in loss_dict.items():
            k = k.replace('loss_', '')
            k = f'loss/test/{k}'
            renamed_loss_dict[k] = v

        storage.put_scalars(**renamed_loss_dict, **self.time_data(), smoothing_hint=False)

        return float(losses.detach().cpu().item())
