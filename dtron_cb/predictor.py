import os
from typing import List, Tuple
import json

import cv2
import torch
from tqdm import tqdm
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from .utils.today import today
from .utils.ensure_dir import ensure_dir
from .config import CfgNode
from .particle import Particle


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


class Image:

    def __init__(self, image_id: int, file_name: str, height: int, width: int, license_idx=0, date_captured='n/a',
                 **kwargs):
        self.idx = image_id
        self.license_idx = license_idx
        self.file_name = file_name
        self.height = height
        self.width = width
        self.date_captured = date_captured
        _ = kwargs  # explicitly do nothing with kwargs

    def dict(self) -> dict:
        return dict(
            id=self.idx,
            license=self.license_idx,
            file_name=self.file_name,
            height=self.height,
            width=self.width,
            date_captured=self.date_captured
        )


class License:

    def __init__(self, url: str, idx: int, name: str):
        self.url = url
        self.idx = idx
        self.name = name

    def dict(self) -> dict:
        return dict(url=self.url, id=self.idx, name=self.name)


class Category:

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name

    def dict(self) -> dict:
        return dict(
            id=self.idx,
            name=self.name
        )


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


class Particles:

    def __init__(self):
        self.particles: List[Particle] = []

    def add(self, image: np.array, contour, px2um):
        self.particles.append(Particle(image, contour, px2um))

    def write_out(self, fn: str, comment=None):
        csv_lines = [','.join(Particle.CSV_HEADER)]
        if comment:
            csv_lines.insert(0, '# ' + comment)
        for particle in self.particles:
            csv_lines.append(particle.to_csv_line())

        with open(fn, 'w') as f:
            for line in csv_lines:
                f.write(f'{line}\n')


class COCOPredictor:

    MAX_N_POLYGON = 25

    def __init__(self, config: CfgNode):
        self.output_dir = config.OUTPUT_DIR
        self.images_dir = ensure_dir(f'{self.output_dir}/segmented_images')

        self.model = build_model(config)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(config.MODEL.WEIGHTS)

        self.datasets = config.DATASETS.NAMES
        self.datasets_root = config.DATASETS.ROOT

        self.crop = config.DATA.CROP

    def predict(self):
        for ds_name in self.datasets:
            self._predict_dataset(ds_name)

    def _predict_dataset(self, ds_name: str):
        ds = DatasetCatalog.get(ds_name)
        md = MetadataCatalog.get(ds_name)

        annotated_dataset = COCO_Dataset()
        did2cid = md.get('thing_dataset_id_to_contiguous_id')
        cid2did = {v: k for k, v in did2cid.items()}
        for cid, cat in enumerate(md.get('thing_classes')):
            did = cid2did[cid]
            annotated_dataset.categories.append(Category(did, cat))

        particles = Particles()

        for d in tqdm(ds):
            d = dict(**d)
            fn = d['file_name']
            del d['file_name']

            imdata = Image(file_name=os.path.relpath(fn, self.datasets_root), **d)
            annotated_dataset.images.append(imdata)

            oim = cv2.imread(fn)
            oimc = oim.copy()

            if self.crop:
                x1, y1, x2, y2 = self.crop
                oim = oim[y1:y2, x1:x2]

            im_ident = fn.replace('/', '-').replace('\\', '-')
            # im = cv2.resize(im, (512, 512))
            # im = torch.as_tensor(oim.astype('float32').transpose(2, 0, 1))
            im = torch.as_tensor(oim.astype('float32')); im = im.permute((2, 0, 1))
            inputs = [dict(image=im)]
            instances = self.model.inference(inputs, do_postprocess=True)[0]['instances']

            try:
                scores = instances.scores
                bboxes = instances.pred_boxes
                cats = instances.pred_classes
                masks = instances.pred_masks
            except AttributeError:
                print(list(instances.get_fields().keys()))
                raise
            for score, bbox, cat, mask in zip(scores, bboxes, cats, masks):
                if score < 0.95:
                    continue

                x1, y1, x2, y2 = bbox
                bbox = x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                area = w*h
                mask = (mask.cpu().numpy()*255).astype(np.uint8)
                cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                # if multiple contours are found; just take the largest
                if len(cnt) > 1:
                    cnt = sorted(cnt, key=lambda c: cv2.contourArea(c))[-1]
                else:
                    cnt = cnt[0]
                cnt = cv2.convexHull(cnt)
                npoints = len(cnt)
                fac = int(npoints / self.MAX_N_POLYGON)
                if fac > 1:
                    # extra array constructor is required, opencv is funny about contours
                    cnt = np.array(cnt[::fac], dtype=np.int32)

                cv2.drawContours(oimc, [cnt], 0, (0, 255, 255), 2)

                particles.add(oimc, cnt[0], 1)  # TODO get px2um

                # convert [[[x, y]], ... ] format to [x, y, x, y, ...]
                cnt = np.array(cnt)
                cnt = cnt.squeeze()
                seg = np.zeros(cnt.size)
                seg[::2] = cnt[:, 0]
                seg[1::2] = cnt[:, 1]
                seg = [[int(s) for s in seg]]

                anndata = Annotation(
                    idx=len(annotated_dataset.annotations),
                    image_idx=int(imdata.idx),
                    category_idx=cid2did[int(cat)],
                    bbox=bbox,
                    segmentation=seg,
                    area=area
                )
                annotated_dataset.annotations.append(anndata)

            # write out segmented image
            cv2.imwrite(f'{self.images_dir}/{im_ident}', oimc)

        annotated_dataset.write_out(f'{self.output_dir}/annot_{today}_{ds_name}.json')

        particles.write_out(f'{self.output_dir}/particles.csv', comment='lengths are in pixels')

        print(len(annotated_dataset.annotations))
