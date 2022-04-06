from typing import Tuple, List


Int4 = Tuple[int, int, int, int]


class Annotation:

    def __init__(self, idx: int, image_idx: int, category_idx: int, bbox: Int4, segmentation: List[List[float]], area: float):
        self.idx = idx
        self.image_idx = image_idx
        self.category_idx = category_idx
        self.bbox = bbox
        self.segmentation = segmentation
        self.area = area

    def dict(self) -> dict:
        return dict(
            id=self.idx,
            image_id=self.image_idx,
            category_id=self.category_idx,
            bbox=self.bbox,
            segmentation=self.segmentation,
            area=self.area,
            iscrowd=0
        )
