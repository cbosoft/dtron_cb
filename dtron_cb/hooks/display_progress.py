from tqdm import tqdm, trange

from .base import HookBase


class DisplayProgressHook(HookBase):

    def __init__(self):
        self.bar: tqdm = None
        self.total_train_loss = self.total_test_loss = 0.0
        self.n_train_batches = self.n_test_batches = 0

    def before_train(self):
        self.bar = trange(self.trainer.n_epochs, unit='epochs')

    def after_train_batch(self):
        self.total_train_loss += self.trainer.this_batch_train_loss
        self.n_train_batches += 1

    def after_test_batch(self):
        self.total_test_loss += self.trainer.this_batch_test_loss
        self.n_test_batches += 1

    def after_epoch(self):
        mean_train_loss = self.total_train_loss / self.n_train_batches
        mean_test_loss = self.total_test_loss / self.n_test_batches
        self.total_train_loss = self.total_test_loss = 0.0
        self.n_train_batches = self.n_test_batches = 0

        self.bar.set_description(f'{mean_train_loss:.3e} ({mean_test_loss:.3e})', False)
        self.bar.update()
