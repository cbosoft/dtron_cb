import os.path
from glob import glob

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from .config import CfgNode


def register_test_train(root_path: str, json_file_path: str, train_frac=0.8, test_frac=0.0, strip_empty=True):
    name, _ = os.path.splitext(os.path.basename(json_file_path))

    all_data = load_coco_json(json_file_path, root_path, name)
    if strip_empty:
        all_data = [d for d in all_data if d['annotations']]
    n = len(all_data)
    n_test = int(n*test_frac)
    n_train_valid = n - n_test
    n_train = int(n_train_valid*train_frac)
    np.random.shuffle(all_data)

    test_data = all_data[:n_test]
    train_data = all_data[n_test:n_train]
    valid_data = all_data[n_train:]

    # All
    DatasetCatalog.register(name, lambda: all_data)
    MetadataCatalog.get(name).set(json_file=json_file_path, image_root=root_path, evaluator_type="coco")

    if n_test:
        DatasetCatalog.register(name+'_test', lambda: test_data)
        MetadataCatalog.get(name+'_test').set(json_file=json_file_path, image_root=root_path, evaluator_type="coco")

    # Train
    DatasetCatalog.register(name+'_train', lambda: train_data)
    MetadataCatalog.get(name+'_train').set(json_file=json_file_path, image_root=root_path, evaluator_type="coco")

    DatasetCatalog.register(name+'_valid', lambda: valid_data)
    MetadataCatalog.get(name+'_valid').set(json_file=json_file_path, image_root=root_path, evaluator_type="coco")


def register_datasets(config: CfgNode):
    datasets_root = config.DATASETS.ROOT
    train_frac = config.DATASETS.TRAIN_FRACTION
    test_frac = config.DATASETS.TEST_FRACTION
    datasets_files = glob(f'{datasets_root}/*.json')

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    for dsf in datasets_files:
        register_test_train(datasets_root, dsf, train_frac=train_frac, test_frac=test_frac,
                            strip_empty=config.DATALOADER.FILTER_EMPTY_ANNOTATIONS)

    if config.ACTION in ('train', 'xval'):
        number_batches = sum([len(DatasetCatalog.get(dsname)) for dsname in config.DATASETS.TRAIN])
        config.SOLVER.MAX_ITER = config.SOLVER.N_EPOCHS * number_batches
    else:
        config.SOLVER.MAX_ITER = -1
