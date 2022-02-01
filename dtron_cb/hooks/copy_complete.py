import shutil

from detectron2.engine.hooks import HookBase

from ..utils.today import today


class CopyCompleteHook(HookBase):

    def __init__(self, cfg):
        self.src = cfg.OUTPUT_DIR
        self.dest = cfg.EXPERIMENTS_META.FINAL_ROOT + '/' + today

    def after_train(self):
        print('Copying results to backup dir')
        print(f'effectively: cp -r "{self.src}" "{self.dest}"')
        shutil.copytree(self.src, self.dest)

