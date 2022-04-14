from collections import defaultdict
from typing import List, Optional
import logging
import weakref

from tqdm import trange
import torch
from detectron2.utils.events import EventStorage

from ..hooks.base import Hooks, HookBase, D2_HookBase


class TrainerBase:

    def __init__(self, n_epochs, train_loader, valid_loader, test_loader):
        self.n_epochs = n_epochs
        self.epoch = 0
        self.total_train_batches = 0
        self.total_valid_batches = 0
        self.total_test_batches = 0
        self.hooks = Hooks()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.storage: Optional[EventStorage] = None
        self.this_batch_train_loss = 0.0
        self.this_batch_valid_loss = 0.0
        self.this_batch_test_loss = 0.0
        self.state = 'pre'

    @property
    def iter(self) -> int:
        return self.total_train_batches

    @property
    def start_iter(self) -> int:
        return 0

    @property
    def max_iter(self) -> int:
        return self.n_epochs*len(self.train_loader)

    def register_hooks(self, hooks: List[Optional[D2_HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, D2_HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        self.state = 'training'
        logger = logging.getLogger(__name__)
        with EventStorage(0) as self.storage:
            try:
                self.hooks.before_train()
                for self.epoch in range(self.n_epochs):
                    # Train:
                    for batch in self.train_loader:
                        self.hooks.before_train_batch()
                        self.this_batch_train_loss = self.do_train_batch(batch)
                        self.total_train_batches += 1
                        self.storage.step()
                        self.hooks.after_train_batch()

                    # Validation
                    with torch.no_grad():
                        for batch in self.valid_loader:
                            self.hooks.before_valid_batch()
                            self.this_batch_valid_loss = self.do_valid_batch(batch)
                            self.total_valid_batches += 1
                            self.storage.step()
                            self.hooks.after_valid_batch()

                    self.hooks.after_epoch()
            except Exception:
                self.state = 'failed'
                logger.exception("Exception during training:")
                raise
            finally:
                self.hooks.after_train()

    def state_dict(self):
        ret = {"epoch #": self.epoch}
        hooks_state = {}
        for h in self.hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.epoch = state_dict["epoch #"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self.hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")

    def time_data(self) -> dict:
        return dict(epoch=self.epoch, train_batch=self.total_train_batches, test_batch=self.total_test_batches)

    def do_train_batch(self, batch) -> float:
        raise NotImplementedError

    def do_valid_batch(self, batch) -> float:
        raise NotImplementedError

    def do_test_batch(self, batch) -> float:
        raise NotImplementedError
