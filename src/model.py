import torch, torch.nn as nn, torch.nn.functional as F
from typing import Any
from torch import Tensor
from lightning import LightningModule
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


def llr_to_bit(x: Tensor) -> Tensor:
    return (x < 0).to(torch.int8)


class SBNDLitModule(LightningModule):

    def __init__(
        self,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        llr_scaling: float = 2.0,
        learn_llr_scaling: bool = False,
        code_path: str = "",
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.llr_scaling = torch.nn.Parameter(torch.tensor(llr_scaling))
        if not learn_llr_scaling:
            self.llr_scaling.requires_grad = False

        # save all hyper-params required to load a checkpoint
        # decoder is first excluded, and then manually added to avoid lightning's parser warning:
        # "Attribute 'decoder' is an instance of `nn.Module` ..."
        self.save_hyperparameters(logger=False, ignore=["decoder"])
        self.hparams.decoder = decoder  # type: ignore[attr-defined]

        # special attribute required for i/o shapes calculation in model summary
        self.example_input_array = decoder.example_input_array  # type: ignore[assignment]

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        return self.decoder(ym, s) * self.llr_scaling

    def _cw_loss(self, e_pred: Tensor, e_true: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            -e_pred, e_true.float()
        )  # prob(e=1) = sigmoid(-e_pred)

    def _cw_accuracy(self, e_pred: Tensor, e_true: Tensor) -> Tensor:
        e_pred_bin = llr_to_bit(e_pred)
        # only look for errors within predictions for the non-zero target patterns
        is_equal = torch.ones(e_true.size(0), dtype=torch.bool, device=e_true.device)
        nz_target_idx = torch.any(e_true, dim=1).nonzero().squeeze(1)
        is_equal[nz_target_idx] = torch.all(
            e_pred_bin[nz_target_idx] == e_true[nz_target_idx], dim=1
        )
        return torch.mean(is_equal.float()).detach()

    def model_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        ym, s, e_true = batch
        e_pred = self(ym, s)
        e_true_resized = e_true.narrow(
            1, 0, e_pred.size(1)
        )  # make sure target matches model output second dim (n or k)
        loss, acc = self._cw_loss(e_pred, e_true_resized), self._cw_accuracy(
            e_pred, e_true_resized
        )
        return loss, acc

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss, acc = self.model_step(batch)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/acc",
            acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/err", 1 - acc, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        loss, acc = self.model_step(batch)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/acc", acc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True
        )
        self.log("val/err", 1 - acc, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, acc = self.model_step(batch)
        suffix = f"/{dataloader_idx}"
        self.log(
            "test/loss" + suffix,
            loss,
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/acc" + suffix,
            acc,
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/err" + suffix,
            1 - acc,
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def predict_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        ym, s, e_true = batch
        e_pred = self(ym, s)
        e_true_resized = e_true.narrow(
            1, 0, e_pred.size(1)
        )  # make sure target matches model output size
        return e_pred, e_true_resized, s  # preds, targets, syndromes

    def on_train_epoch_start(self) -> None:
        # log learning rate at the start of each epoch (more convenient than LearningRateMonitor cb)
        cur_lr = self.optimizers().param_groups[0]["lr"]  # type: ignore[union-attr]
        self.log("train/lr", cur_lr, sync_dist=True)
        self.log("train/epoch", self.current_epoch, sync_dist=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore[attr-defined]
        if self.hparams.lr_scheduler is None:  # type: ignore[attr-defined]
            return {"optimizer": optimizer}

        scheduler = self.hparams.lr_scheduler(optimizer)  # type: ignore[attr-defined]
        lr_scheduler_config = {"scheduler": scheduler}

        # some schedulers require extra config params, so let's handle them here
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # automatically adjust OneCycleLR's total steps to the estimated number of batches
            stepping_batches = self.trainer.estimated_stepping_batches
            lr_scheduler_config["scheduler"] = self.hparams.lr_scheduler(  # type: ignore[attr-defined]
                optimizer, total_steps=stepping_batches
            )
            lr_scheduler_config["interval"] = "step"
            log.info(f"OneCycleLR will use a total number of {stepping_batches} steps")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler_config["monitor"] = "val/loss"
            log.info(f"ReduceLROnPlateau will monitor {lr_scheduler_config["monitor"]}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


if __name__ == "__main__":
    pass
