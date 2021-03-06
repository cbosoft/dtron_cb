import json

from detectron2.engine.hooks import HookBase

from ..config import CfgNode


class WriteMetaHook(HookBase):

    def __init__(self, cfg: CfgNode, model):
        self.output_dir = cfg.OUTPUT_DIR
        self.pretrained = 'No' if cfg.MODEL.WEIGHTS == 0 else f'Yes: {cfg.MODEL.WEIGHTS}'
        self.config_text = cfg.dump()
        self.datasets = ','.join([*cfg.DATASETS.TRAIN, *cfg.DATASETS.TEST])
        self.model_text = repr(model)

    def before_train(self):
        print('Writing metadata')
        metadata = dict(
            path=f'{self.output_dir}/model_final.pth',
            pretrained=self.pretrained,
            trained_on=self.datasets,
            notes=f'<auto generated by {self.__class__.__name__}>'
        )
        with open(f'{self.output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        with open(f'{self.output_dir}/config.yaml', 'w') as f:
            f.write(self.config_text)

        with open(f'{self.output_dir}/model.txt', 'w') as f:
            f.write(self.model_text)

        self.write_listing(self.trainer.train_loader, 'train')
        self.write_listing(self.trainer.valid_loader, 'valid')
        if self.trainer.test_loader is not None:
            self.write_listing(self.trainer.test_loader, 'test')

    def write_listing(self, dl, name):
        fns = []
        for b in dl:
            for bi in b:
                fns.append(bi['file_name'])
        with open(f'{self.output_dir}/listing_{name}.txt', 'w') as f:
            for fn in fns:
                f.write(f'{fn}\n')
