import numpy as np

from detectron2.structures import Instances
from detectron2.structures.masks import BitMasks
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

__all__ = ['instances_to_mask']


def instances_to_mask(instances: Instances, **kwargs):
    if instances.has('pred_boxes'):
        return _pred_instances_to_mask(instances, **kwargs)
    elif instances.has('gt_masks'):
        return _gt_instances_to_mask(instances)
    raise ValueError('instances does not contain enough information to yield a mask')


def _gt_instances_to_mask(instances: Instances):
    print(instances.get_fields().keys())
    combined = np.zeros(instances.image_size, dtype=bool)
    masks = BitMasks.from_polygon_masks(instances.gt_masks, *instances.image_size)
    for mask in masks.tensor:
        mask_bool = mask.detach().cpu().numpy() > 0.5
        combined |= mask_bool
    return combined


def _pred_instances_to_mask(instances: Instances, *, score_thresh, mask_thresh=None):
    combined = np.zeros(instances.image_size, dtype=bool)
    if mask_thresh is None:
        mask_thresh = GeneralizedRCNN.mask_threshold
    try:
        for score, mask in zip(instances.scores, instances.pred_masks):
            if score < score_thresh: continue
            mask_bool = mask.detach().cpu().numpy() > mask_thresh*255
            combined |= mask_bool
    except AttributeError:
        pass

    return combined