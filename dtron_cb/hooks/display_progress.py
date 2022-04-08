from tqdm import tqdm, trange

from .base import HookBase


class DisplayProgressHook(HookBase):

    def __init__(self):
        self.bar: tqdm = None
        self.total_train_loss = self.total_valid_loss = 0.0
        self.n_train_batches = self.n_valid_batches = 0

    def before_train(self):
        self.bar = trange(self.trainer.n_epochs, unit='epochs', ncols=20)

    def after_train_batch(self):
        self.total_train_loss += self.trainer.this_batch_valid_loss
        self.n_train_batches += 1

    def after_valid_batch(self):
        self.total_valid_loss += self.trainer.this_batch_valid_loss
        self.n_valid_batches += 1

    def after_epoch(self):
        mean_train_loss = self.total_train_loss / self.n_train_batches
        mean_valid_loss = self.total_valid_loss / self.n_valid_batches
        self.total_train_loss = self.total_valid_loss = 0.0
        self.n_train_batches = self.n_valid_batches = 0

        self.bar.set_description(f'{mean_train_loss:.3e} ({mean_valid_loss:.3e})', False)
        self.bar.update()
