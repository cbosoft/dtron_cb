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
    combined = np.zeros(instances.image_size, dtype=bool)
    masks = BitMasks.from_polygon_masks(instances.gt_masks, *instances.image_size)
    for mask in masks.tensor:
        mask_bool = mask.detach().cpu().numpy() > 0.5
        combined |= mask_bool
    return combined


def _pred_instances_to_mask(instances: Instances, *, score_thresh):
    combined = np.zeros(instances.image_size, dtype=bool)

    try:
        for score, mask_bool in zip(instances.scores, instances.pred_masks):
            if score < score_thresh: continue
            combined |= mask_bool.detach().cpu().numpy()
    except AttributeError:
        pass

    return combined
