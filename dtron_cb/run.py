from .config import read_config_file
from .trainer import Trainer
from .predictor import COCOPredictor
from .cross_validator import CrossValidator


def _run_training(cfg, filename):
    print(f'Running experiment (training): "{filename}"')
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def _run_inference(cfg, filename):
    print(f'Running inference: "{filename}"')
    predictor = COCOPredictor(cfg)
    predictor.predict()


def _run_xval(cfg, filename):
    print(f'Running cross validation: "{filename}"')
    xval = CrossValidator(cfg)
    xval.cross_validate()


def run(filename: str):
    cfg = read_config_file(filename)

    actions = dict(
        train=_run_training,
        predict=_run_inference,
        cross_validate=_run_xval
    )

    if cfg.ACTION in actions:
        actions[cfg.ACTION](cfg, filename)
    else:
        raise ValueError(f'config.ACTION must be one of {list(actions.keys())}.')
