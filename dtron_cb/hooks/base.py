from typing import List

from detectron2.engine.train_loop import HookBase as D2_HookBase


class HookBase(D2_HookBase):
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for epoch in range(n_epochs):
            for batch in train_dl:
                hook.before_train_batch()
                trainer.run_batch()
                hook.after_train_batch()
            with torch.no_grad():
                for batch in test_dl:
                    hook.before_test_batch()
                    trainer.run_batch()
                    hook.after_test_batch()
            hook.after_epoch()
        hook.after_train()

    This is adapted from the hooks originally available in detectron.
    _D2_HookBAse.before_step and _D2_HookBAse.after_step are depreciated, but interpreted as before and after train batch hooks.
    """

    def before_train_batch(self):
        pass

    def after_train_batch(self):
        pass

    def before_test_batch(self):
        pass

    def after_test_batch(self):
        pass

    def after_epoch(self):
        pass


class Hooks:

    def __init__(self):
        self._hooks: List[HookBase] = []

    def __iter__(self):
        return iter(self._hooks)

    def append(self, other: HookBase):
        self._hooks.append(other)

    def extend(self, others):
        self._hooks.extend(others)

    def before_train(self):
        for hook in self._hooks:
            hook.before_train()

    def after_train(self):
        for hook in self._hooks:
            hook.after_train()

    def before_train_batch(self):
        for hook in self._hooks:
            hook.before_step()
            try:
                hook.before_train_batch()
            except AttributeError:
                continue

    def after_train_batch(self):
        for hook in self._hooks:
            hook.after_step()
            try:
                hook.after_train_batch()
            except AttributeError:
                continue

    def before_test_batch(self):
        for hook in self._hooks:
            try:
                hook.before_test_batch()
            except AttributeError:
                continue

    def after_test_batch(self):
        for hook in self._hooks:
            try:
                hook.after_test_batch()
            except AttributeError:
                continue

    def after_epoch(self):
        for hook in self._hooks:
            try:
                hook.after_epoch()
            except AttributeError:
                continue
