# The LightningModule wrapper around the SBND decoder model, handling training, 
# validation and testing steps, as well as optimizers and learning rate 
# schedulers configuration. It also includes some monitoring tools to track 
# training dynamics and detect potential issues.

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
        full_monitoring: bool = False,
    ) -> None:
        super().__init__()
        self.full_monitoring = full_monitoring  # if True, will log additional metrics to monitor training dynamics and detect potential issues

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
        loss = self._cw_loss(e_pred, e_true_resized)
        acc = self._cw_accuracy(e_pred, e_true_resized)
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
            log.info(f"ReduceLROnPlateau will monitor {lr_scheduler_config['monitor']}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # monitor gradient norms to detect potential exploding gradients issues
        norms = [
            p.grad.detach().norm()
            for p in self.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if norms:
            self._grad_norm_acc.append(torch.stack(norms).norm().item())

        # monitor Adam effective steps to detect potential underflow issues,
        # but only every 50 steps to minimize overhead.
        if self.global_step % 50 == 0:
            effective_steps = [
                (
                    optimizer.state[p]["exp_avg"]
                    / (optimizer.state[p]["exp_avg_sq"].sqrt() + group.get("eps", 1e-8))
                )
                .abs()
                .max()
                .item()
                for group in optimizer.param_groups
                for p in group["params"]
                if p.requires_grad
                and p in optimizer.state
                and "exp_avg" in optimizer.state[p]
                and "exp_avg_sq" in optimizer.state[p]
            ]
            if effective_steps:
                self._adam_step_max_acc.append(max(effective_steps))

    def on_train_epoch_start(self) -> None:
        # log learning rate at the start of each epoch (more convenient than LearningRateMonitor cb)
        cur_lr = self.optimizers().param_groups[0]["lr"]  # type: ignore[union-attr]
        self.log("train/lr", cur_lr, sync_dist=True)
        self.log("train/epoch", self.current_epoch, sync_dist=True)
        # reset monitoring accumulators at the start of each epoch
        self._grad_norm_acc: list[float] = []
        self._adam_step_max_acc: list[float] = []

    def on_train_epoch_end(self) -> None:
        # log cumulated layer norms
        layer_norms = {
            name: p.detach().norm().item() for name, p in self.named_parameters()
        }
        metrics = {"cum_weight_norm": sum(layer_norms.values())}

        # log gradient stats
        if self._grad_norm_acc:
            metrics["grad_norm_mean"] = sum(self._grad_norm_acc) / len(
                self._grad_norm_acc
            )
            metrics["grad_norm_max"] = max(self._grad_norm_acc)

        # log Adam v min stats to monitor potential underflow issues
        if self._adam_step_max_acc:
            metrics["adam_effective_step_max"] = max(self._adam_step_max_acc)

        # also log norm of each layer parameters if full monitoring is enabled
        if self.full_monitoring:
            metrics.update({f"layers_norms/{k}": v for k, v in layer_norms.items()})

        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

    def on_train_batch_end(
        self, outputs: Any, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        loss = outputs["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"[step {self.global_step}] Invalid loss: {loss.item()}")

        # check for NaN/Inf values in model parameters every 50 steps
        if self.full_monitoring and batch_idx % 50 == 0:
            all_params = torch.cat([p.detach().flatten() for p in self.parameters()])
            if not torch.isfinite(all_params).all():
                # localize which parameters contain NaN/Inf values
                for name, p in self.named_parameters():
                    if not torch.isfinite(p).all():
                        raise ValueError(
                            f"[step {self.global_step}] NaN/Inf found in {name}"
                        )


if __name__ == "__main__":
    pass
