import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog

from .trainer import Trainer
from .config import CfgNode
from .utils.ensure_dir import ensure_dir


class CrossValidator:

    def __init__(self, config: CfgNode):
        self.config = config
        assert config.DATASETS.NAMES is not None, 'config.DATASETS.NAMES must be specified. '\
                                                  'Test or train sets cannot be specified ' \
                                                  'explicitly for cross validation.'

    @staticmethod
    def register_fold(i: int, is_test: bool, fold_data):
        pfx = 'test' if is_test else 'train'
        name = f'{pfx}_fold_{i}'
        DatasetCatalog.register(name, lambda: fold_data)
        MetadataCatalog.get(name).set(evaluator_type="coco", thing_classes=('particles', 'needles', 'agglomerates'))

    def cross_validate(self):
        all_data = []
        for name in self.config.DATASETS.NAMES:
            all_data.extend(DatasetCatalog.get(name))

        n_folds = self.config.CROSS_VALIDATION.N_FOLDS
        np.random.shuffle(all_data)
        n = len(all_data)
        max_n = n - (n % n_folds)
        indices = np.arange(max_n)
        folds = np.split(indices, n_folds)

        folds_output_dir = f'{self.config.OUTPUT_DIR}/fold_{{}}'

        for i in range(n_folds):
            train_indices = np.array(folds[1:]).flatten()
            test_indices = np.array(folds[0]).flatten()
            folds = np.roll(folds, 1)

            train_data = [all_data[idx] for idx in train_indices]
            test_data = [all_data[idx] for idx in test_indices]

            self.register_fold(i + 1, False, train_data)
            self.register_fold(i + 1, True, test_data)

            subconfig = self.config.clone()
            subconfig.defrost()
            subconfig.DATASETS.NAMES = []
            subconfig.DATASETS.TRAIN = [f'train_fold_{i+1}']
            subconfig.DATASETS.TEST = [f'test_fold_{i+1}']
            subconfig.OUTPUT_DIR = ensure_dir(folds_output_dir.format(i+1))
            subconfig.freeze()

            trainer = Trainer(subconfig)
            trainer.resume_or_load(False)
            trainer.train()
