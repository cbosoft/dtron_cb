from .config import read_config_file
from .trainer import Trainer
# from .predictor import Predictor


def _run_training(cfg, filename):
    print(f'Running experiment (training): "{filename}"')
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def _run_inference(cfg, filename):
    print(f'Running inference: "{filename}"')
    predictor = Predictor(cfg)
    predictor.predict()


def run(filename: str):
    cfg = read_config_file(filename)

    actions = dict(
        train=_run_training,
        predict=_run_inference)

    if cfg.ACTION in actions:
        actions[cfg.ACTION](cfg, filename)
    else:
        raise ValueError(f'config.ACTION must be one of {list(actions.keys())}.')
