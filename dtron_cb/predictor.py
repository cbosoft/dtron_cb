import os
from typing import List, Tuple, Dict
import json

import cv2
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from .utils.today import today
from .utils.ensure_dir import ensure_dir
from .config import CfgNode
from .particle import Particle, ParticleConstructionError


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

    def __init__(self, *particles):
        self.particles: List[Particle] = [*particles]

    def add(self, fn: str, orig_image: np.ndarray, contour, px2um, score):
        self.particles.append(Particle(fn, orig_image, contour, px2um, score))

    def write_out(self, fn: str, comment=None):
        csv_lines = [','.join(Particle.CSV_HEADER)]
        if comment:
            csv_lines.insert(0, '# ' + comment)
        for particle in sorted(self.particles):
            csv_lines.append(particle.to_csv_line())

        with open(fn, 'w') as f:
            for line in csv_lines:
                f.write(f'{line}\n')

    def split_by_dir(self) -> Dict[str, "Particles"]:
        by_dir = dict()
        for p in self.particles:
            fn = p.image_file_name
            dn = os.path.dirname(fn)
            if dn not in by_dir:
                by_dir[dn] = list()
            by_dir[dn].append(p)

        return {k: Particles(*sorted(v)) for k, v in by_dir.items()}

    def split_by_fn_chunks(self, fnss: List[str]):
        chunks = [list() for _ in fnss]
        for i, fns in enumerate(fnss):
            for p in self.particles:
                if p.image_file_name in fns:
                    chunks[i].append(p)
        return chunks

    def to_dict(self) -> Dict[str, list]:
        values = {k: list() for k in Particle.CSV_HEADER}
        for p in self.particles:
            d = p.to_dict()
            for k in Particle.CSV_HEADER:
                values[k].append(d[k])
        return values


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
        self.px2um = config.INFERENCE.PX_TO_UM
        self.px_thresh = config.INFERENCE.PIXEL_THRESH
        self.overall_thresh = config.INFERENCE.OVERALL_THRESH

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
            n = 0

            if self.crop:
                x1, y1, x2, y2 = self.crop
                oim = oim[y1:y2, x1:x2]

            im_ident = fn.replace('/', '-').replace('\\', '-')[:-4]
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

            composite = cv2.cvtColor(oim, cv2.COLOR_BGR2RGB)
            for i, (mask_t, score, bbox, cat) in enumerate(zip(masks, scores, bboxes, cats)):

                x1, y1, x2, y2 = bbox
                bbox = x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                area = w*h
                score = float(score.cpu())
                if score < self.overall_thresh:
                    continue

                mask = (mask_t.cpu().numpy()*255).astype(np.uint8)
                cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                # if multiple contours are found; just take the largest
                if len(cnt) > 1:
                    cnt = sorted(cnt, key=lambda c: cv2.contourArea(c))[-1]
                else:
                    cnt = cnt[0]

                try:
                    particles.add(fn, oimc, cnt, self.px2um, float(score))
                except ParticleConstructionError as e:
                    print(f'Not adding particle: {e}')
                    continue

                c = [ci*255 for ci in plt.cm.viridis(score)[:3]]
                cv2.drawContours(composite, [cnt], 0, c, 2)
                n += 1

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

                submask = (plt.cm.viridis(mask)*255).astype(np.uint8)
                alpha_s = np.where(mask < 5, 0.0, 0.3)
                alpha_c = 1.0 - alpha_s
                for c in range(3):
                    composite[:, :, c] = submask[:, :, c] * alpha_s + composite[:, :, c] * alpha_c
            composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{self.images_dir}/{im_ident}_n={n}.png', composite)


        annotated_dataset.write_out(f'{self.output_dir}/annot_{today()}_{ds_name}.json')

        particles.write_out(f'{self.output_dir}/particles.csv', comment='lengths are in pixels')

        self.plot_particles(particles)

        particles_by_dir = particles.split_by_dir()
        for dn, ps in particles_by_dir.items():
            self.plot_n_particles_dynamic(ps, dn.replace('/', '-').replace('\\', '-'))

        print(len(annotated_dataset.annotations))

    def plot_particles(self, particles, tag=''):
        if isinstance(particles, Particles):
            particles = particles.particles
        keys = Particle.CSV_HEADER
        dicts = [p.to_dict() for p in particles]
        values = {k: [d[k] for d in dicts] for k in keys}
        plot_specs = [
            ('length', 'width'),
            ('length', 'circularity'),
            ('width', 'focus_GDER'),
            ('focus_GDER', 'convexity'),
            ('length', 'aspect_ratio'),
            ('aspect_ratio', 'area')
        ]

        for xlbl, ylbl in plot_specs:
            x = values[xlbl]
            y = values[ylbl]
            xunit = Particle.unit_of(xlbl)
            yunit = Particle.unit_of(ylbl)
            plt.figure()
            plt.plot(x, y, 'o', alpha=0.5)
            plt.xlabel(f'{xlbl} [{xunit}]')
            plt.ylabel(f'{ylbl} [{yunit}]')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/fig_{ylbl}_v_{xlbl}{tag}.pdf')
            plt.close()

    def get_fn_chunks(self):
        fns = []
        for dsname in self.datasets:
            fns.extend([d['file_name'] for d in DatasetCatalog.get(dsname)])
        fns = sorted(fns)

        w = 20
        fn_chunks = [fns[i:i+w] for i in range(0, len(fns), w)]
        return fn_chunks

    def plot_n_particles_dynamic(self, particles, tag):
        chunks = self.get_fn_chunks()
        particless = particles.split_by_fn_chunks(chunks)

        x = np.arange(len(chunks))
        y = np.zeros_like(x)
        for i, particles in enumerate(particless):
            y[i] = len(particles)

        max_ticks = 10
        xtl = ['...'+c[0][-10:] for c in chunks][::len(x)//max_ticks]
        xt = x[::len(x)//max_ticks]

        plt.figure()
        plt.plot(x, y)
        plt.xticks(xt, xtl, rotation=45, ha='right')
        plt.ylabel('Particle count')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig_particle_count{tag}.pdf')
        plt.close()
