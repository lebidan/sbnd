# Some useful learning rate schedulers for SBND training

import math
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


class CosineWarmupLR(LRScheduler):
    """CosineAnnealingLR with an added linear warmup. A classic for transformer training"""

    def __init__(
        self, optimizer: Optimizer, warmup: int, max_iters: int, lr_min: float = 0.0
    ) -> None:
        self.warmup, self.lr_min = warmup, lr_min
        self.period = max_iters - warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float | Tensor]:
        epoch = 1 if self.last_epoch < 1 else self.last_epoch - 1
        lr_factor = self.get_lr_factor(epoch)
        return [
            self.lr_min + (base_lr - self.lr_min) * lr_factor
            for base_lr in self.base_lrs
        ]

    def get_lr_factor(self, epoch: int) -> float:
        if epoch < self.warmup:
            lr_factor = epoch * 1.0 / self.warmup
        else:
            lr_factor = 0.5 * (
                1 + math.cos(math.pi * (epoch - self.warmup) / self.period)
            )
        return lr_factor


class WarmupStableDecayLR(LambdaLR):
    """The modern standard for training deep learning models, see https://arxiv.org/abs/2405.18392"""

    def __init__(
        self,
        optimizer: Optimizer,
        total: int,
        warmup: int = 5,
        decay: int = 0,
        last_epoch: int = -1,
    ) -> None:
        lr_factor_fun = lambda epoch: self._wsd_lr_factor(epoch, total, warmup, decay)
        super().__init__(optimizer, lr_factor_fun, last_epoch)

    @staticmethod
    def _wsd_lr_factor(epoch: int, total: int, warmup: int, decay: int) -> float:
        assert (warmup + decay) <= total
        if epoch < warmup:
            alpha = (epoch + 1) / warmup
            return min(alpha, 1)
        elif epoch < total - decay:
            return 1.0
        else:
            alpha = 1.0 if decay == 0 else (total - epoch) / decay
            return max(1 - math.sqrt(1 - alpha), 0)


if __name__ == "__main__":
    pass
