from .config import read_config_file
from .trainer import Trainer
from .predictor import COCOPredictor
from .cross_validator import CrossValidator
from .utils.today import tick


def _run_training(cfg, filename, n=5):
    print(f'Running experiment (training): "{filename}"')

    try:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except FloatingPointError as e:
        print(f'Training failed: {e}')
        if n > 0:
            print(f'Trying again ({n-1} attempts remain)')
            _run_training(cfg, filename, n-1)
        else:
            print('Have reached the maximum number of retries: giving up.')
            print('Try reducing CONFIG.SOLVER.BASE_LR, or perhaps CONFIG.SOLVER.WARMUP_ITERS')


def _run_inference(cfg, filename):
    print(f'Running inference: "{filename}"')
    predictor = COCOPredictor(cfg)
    predictor.predict()


def _run_xval(cfg, filename):
    print(f'Running cross validation: "{filename}"')
    xval = CrossValidator(cfg)
    xval.cross_validate()


def run(filename: str):
    tick()
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
