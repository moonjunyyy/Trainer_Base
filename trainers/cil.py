from .trainer import Trainer

class CIL(Trainer):
    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)
        self.num_tasks = kargs.get('num_tasks', 1)
        self.reset_optimizer = kargs.get('reset_optimizer', False)

    def setup(self, *args, **kargs):
        pass