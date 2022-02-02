from typing import List
import yaml
import random

import torch
import numpy as np
from torch.cuda import is_available as is_cuda_available
from detectron2.config import CfgNode, get_cfg as _get_root_config
from detectron2.model_zoo import get_config_file as get_zoo_config

from .utils.ensure_dir import ensure_dir
from .utils.today import today
from .datasets import register_datasets


def get_config() -> CfgNode:
    config = _get_root_config()
    apply_defaults(config)
    return config


def apply_defaults(config: CfgNode) -> CfgNode:

    config.ACTION = 'train'
    config.PARENT = None
    config.DATASETS.ROOT = '/media/raid/cboyle/datasets'
    config.DATASETS.NAMES = None
    config.DATASETS.TRAIN = None
    config.DATASETS.TEST = None
    config.DATASETS.TRAIN_FRACTION = 0.8

    config.TEST.EVAL_PERIOD = 100
    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    config.DATALOADER.NUM_WORKERS = 4
    config.MODEL.ROI_HEADS.NUM_CLASSES = 1
    config.MODEL.DEVICE = 'cuda' if is_cuda_available() else 'cpu'

    config.SOLVER.IMS_PER_BATCH = 4
    config.SOLVER.MAX_ITER = 5000
    config.SOLVER.WARMUP_ITERS = 500
    config.SOLVER.CHECKPOINT_PERIOD = 500
    config.SOLVER.STEPS = 4000, 4500

    config.EXPERIMENTS_META = CfgNode()
    config.EXPERIMENTS_META.ROOT = "training_results"
    config.EXPERIMENTS_META.FINAL_ROOT = None
    config.EXPERIMENTS_META.SHOULD_COPY_ROOT = False

    config.DATA = CfgNode()
    config.DATA.AUGMENTATIONS = [
        'T.ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style=\'choice\')',
        'T.RandomFlip()',
        'T.RandomBrightness(0.9, 1.1)',
        'T.RandomContrast(0.9, 1.1)'
    ]
    config.DATA.CROP = None

    return config


def finalise(config: CfgNode):

    assert config.DATASETS.ROOT, f'"config.DATASETS.ROOT" must be specified as the dir containing COCO format .json files.'

    assert 0.1 < config.DATASETS.TRAIN_FRACTION < 0.9

    assert config.EXPERIMENTS_META.ROOT, f'"config.EXPERIMENTS_META.ROOT" must be set with a location to store experiment information while running.'
    config.OUTPUT_DIR = ensure_dir(f'{config.EXPERIMENTS_META.ROOT}/{today}')
    config.EXPERIMENTS_META.SHOULD_COPY_ROOT = config.EXPERIMENTS_META.FINAL_ROOT is not None

    if config.EXPERIMENTS_META.SHOULD_COPY_ROOT:
        ensure_dir(config.EXPERIMENTS_META.FINAL_ROOT)

    if config.ACTION == "train":
        if isinstance(config.DATASETS.NAMES, str):
            names = [config.DATASETS.NAMES]
        else:
            names = config.DATASETS.NAMES
        if not config.DATASETS.TRAIN:
            config.DATASETS.TRAIN = tuple([f'{n}_train' for n in names])
            config.DATASETS.TEST = tuple([f'{n}_test' for n in names])

    if config.SEED is None or config.SEED < 0:
        config.SEED = random.randint(0, 1_000_000)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    config.freeze()

    register_datasets(config)

    return config


def get_config_parents(filename: str) -> List[str]:

    if filename.startswith('zoo:'):
        filename = get_zoo_config(filename[4:])

    with open(filename) as f:
        yamlstr = f.read()

    data = yaml.safe_load(yamlstr)

    if data is not None and 'PARENT' in data and data['PARENT']:
        parent = data['PARENT']
        return [*get_config_parents(parent), parent]
    else:
        return []


def read_config_file(filename: str):

    config = get_config()

    to_merge = [*get_config_parents(filename), filename]
    for i, fn in enumerate(to_merge):
        if fn.startswith('zoo:'):
            assert i == 0, 'Only top level can inherit from zoo.'
            fn = get_zoo_config(fn[4:])
            config.merge_from_file(fn)
            apply_defaults(config)
        else:
            config.merge_from_file(fn)

    return finalise(config)
