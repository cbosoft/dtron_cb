from typing import List
import json

from .category import Category
from .license import License
from .annotation import Annotation
from .image import Image


class COCO_Dataset:

    def __init__(self, year='n/a', version='n/a', description='n/a', contributor='n/a', url='n/a', date_created=None):
        self.info = dict(year=year, version=version, description=description,
                         contributor=contributor, url=url, date_created=date_created)
        self.licenses: List[License] = self.get_default_licenses()
        self.images: List[Image] = []
        self.annotations: List[Annotation] = []
        self.categories: List[Category] = []

    @staticmethod
    def get_default_licenses() -> List[License]:
        return [License('n/a', 0, 'CMAC internal use only')]

    def write_out(self, fn: str):

        data_dict = dict(
            info=self.info,
            licenses=[l.dict() for l in self.licenses],
            images=[i.dict() for i in self.images],
            annotations=[a.dict() for a in self.annotations],
            categories=[c.dict() for c in self.categories]
        )

        with open(fn, 'w') as f:
            json.dump(data_dict, f)
