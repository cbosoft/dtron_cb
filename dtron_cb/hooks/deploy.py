import json

import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.env import TORCH_VERSION
from detectron2.export import TracingAdapter
from detectron2.modeling import GeneralizedRCNN

from ..config import CfgNode


class DeployModelHook(HookBase):

    def __init__(self, cfg: CfgNode, model):
        self.output_dir = cfg.OUTPUT_DIR
        self.model = model

    def after_train(self):
        self.model.eval()
        self.deploy_model(self.model.cpu(), self.output_dir)
        self.deploy_model(self.model.cuda(), self.output_dir)
        self.model.train()

    @staticmethod
    def deploy_model(model, output_dir: str):
        dummy_inputs = [{"image": torch.tensor(0)}]
        assert TORCH_VERSION >= (1, 8)

        if isinstance(model, GeneralizedRCNN):

            def inference(model, inputs):
                # use do_postprocess=False so it returns ROI mask
                inst = model.inference(inputs, do_postprocess=False)[0]
                return [{"instances": inst}]

        else:
            inference = None  # assume that we just call the model directly

        traceable_model = TracingAdapter(model, dummy_inputs, inference)

        is_cuda_model = model.device != torch.device('cpu')
        sufx = '_cuda' if is_cuda_model else ''
        ts_model = torch.jit.trace(traceable_model, (dummy_inputs[0]["image"],))
        oname = f'{output_dir}/detectron2{sufx}.ts'
        with open(oname, 'wb') as f:
            torch.jit.save(ts_model, f)
        t = 'GPU-based' if is_cuda_model else 'CPU-based'
        print(f'Exported {t} model to "{oname}" via tracing method.')
