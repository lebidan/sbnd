import logging, hydra, os
import torch, lightning as lit

from typing import Any, Iterable, cast
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.loggers import Logger

from .utils import get_rank_zero_logger
from .model import SBNDLitModule

log = get_rank_zero_logger(__name__)


def log_config(
    cfg: DictConfig, param_count: int, loggers: Logger | Iterable[Logger] | None = None
) -> None:
    """Log selected configuration settings for the current experiment"""

    if loggers is None:
        return

    # general settings
    hparams = {"seed": cfg.get("seed")}

    # filter and reformat the most relevant settings for the training
    for key in ["code", "data", "decoder", "optimizer", "lr_scheduler", "trainer"]:
        if cfg.get(key) is not None:
            opts = cast(dict[str, Any], OmegaConf.to_container(cfg[key], resolve=True))
            opts.pop("_partial_", None)
            for k in opts.keys():
                hparams[key + "/" + k] = opts[k]

    # also keep track of the total number of trainable model parameters
    hparams["decoder/total_params"] = param_count

    # fetch selected settings to each logger
    if loggers:
        loggers_iter = loggers if not isinstance(loggers, Logger) else [loggers]
        for logger in loggers_iter:
            logger.log_hyperparams(hparams)


def load_pretrained_model(model_path: str, **kwargs) -> SBNDLitModule:
    """Load a lightning checkpoint"""
    return SBNDLitModule.load_from_checkpoint(model_path, weights_only=False, **kwargs)


# set the path to the conf directory, which is one level up from the current file
_conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf")


@hydra.main(version_base="1.3", config_path=_conf_dir, config_name="train")
def main(cfg: DictConfig) -> None:

    # enable TF32 fast ops if available
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")    
    
    # disable torch.dynamo warnings, as dynamo can be *very* verbose
    # see https://github.com/pytorch/pytorch/issues/94788
    torch._logging.set_logs(dynamo=logging.ERROR)

    # set seed for reproducibility (if any)
    if cfg.get("seed"):
        lit.seed_everything(cfg.seed, workers=True, verbose=False)
        log.info(f"Global seed set to {cfg.seed}")

    # analyze the training environment
    if os.environ.get("SLURM_JOB_ID"):
        log.info("Training on SLURM cluster")
        slurm_nodes = (
            os.environ.get("SLURM_NNODES") or
            os.environ.get("SLURM_JOB_NUM_NODES") or
            "?"
        )
        slurm_gpus = (
            os.environ.get("SLURM_GPUS_ON_NODE") or
            os.environ.get("SLURM_GPUS_PER_NODE") or
            os.environ.get("SLURM_JOB_GPUS") or
            "?"
        )
        slurm_tasks = (
            os.environ.get("SLURM_NTASKS_PER_NODE") or
            os.environ.get("SLURM_TASKS_PER_NODE") or
            os.environ.get("SLURM_NTASKS") or
            "?"
        )
        slurm_cpus = (
            os.environ.get("SLURM_CPUS_PER_TASK") or
            os.environ.get("SLURM_CPUS_ON_NODE") or
            os.environ.get("SLURM_JOB_CPUS_PER_NODE") or
            "?"
        )
        log.info(
            f"SLURM environment: {slurm_nodes} node(s), "
            f"{slurm_gpus} GPU(s)/node, {slurm_tasks} task(s)/node, "
            f"{slurm_cpus} CPU(s)/task"
        )
    else:
        log.info("Training on local machine")

    # adapt the training setup in case of multi-gpu training
    total_gpus = cfg.nodes * cfg.gpus
    if total_gpus > 1:
        log.info(
            f"Starting DDP training using {cfg.nodes} node(s) x {cfg.gpus} GPU(s)/node"
        )

        # distribute total batch size in config file across all GPUs
        bs_per_gpu = cfg.data.train_bs // total_gpus
        cfg.data.train_bs = bs_per_gpu  # adjust bs before initializing datamodule
        log.info(f"Total training batch size = {bs_per_gpu * total_gpus}")
        log.info(
            f"Batch size per GPU = {bs_per_gpu}, with {cfg.cpus} workers/GPU for dataloading"
        )

    else:
        log.info(f"Starting single-GPU training")
        log.info(
            f"Batch size = {cfg.data.train_bs}, with {cfg.cpus} workers for dataloading"
        )

    # instantiate code & decoder (decoder won't be used with pretrained model)
    code = hydra.utils.instantiate(cfg.code)
    decoder = hydra.utils.instantiate(cfg.decoder, code=code)

    # instantiate lightning datamodule and module
    dm = hydra.utils.instantiate(cfg.data, code=code)

    # instantiate training logger(s)
    loggers = hydra.utils.instantiate(cfg.loggers)

    # instantiate trainer
    training_args = {"num_nodes": cfg.nodes, "devices": cfg.gpus}
    trainer_cb = hydra.utils.instantiate(cfg.trainer_cb)
    if loggers is None:
        loggers = False
    trainer = lit.Trainer(
        **cfg.trainer, **training_args, logger=loggers, callbacks=trainer_cb
    )

    # Check if there is a checkpoint to resume or continue from
    # if so, setup the model and fit options accordingly
    ckpt_to_resume = cfg.get("resume", None)
    ckpt_to_continue = cfg.get("continue", None)
    ckpt_path = None
    code_path = str(cfg.code.mat_file)
    fit_kwargs = {}
    if ckpt_to_resume is not None:
        log.info(f"Resuming training from checkpoint: {ckpt_to_resume}")
        lm = hydra.utils.instantiate(cfg.model, decoder=decoder, code_path=code_path)
        ckpt_path = ckpt_to_resume
        fit_kwargs = {"weights_only": False}
    elif ckpt_to_continue is not None:
        log.info(f"Loading pretrained model from checkpoint: {ckpt_to_continue}")
        lm = load_pretrained_model(
            ckpt_to_continue,
            optimizer=hydra.utils.instantiate(cfg.optimizer),
            lr_scheduler=hydra.utils.instantiate(cfg.lr_scheduler),
            code_path=code_path,
        )
        log.info(f"LLR scaling factor before fit: {lm.llr_scaling.detach().item():.4f}")
    else:
        lm = hydra.utils.instantiate(cfg.model, decoder=decoder, code_path=code_path)

    # keep a record of current experiment configuration, including the number of trainable params
    param_count = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    log_config(cfg, param_count, loggers)

    # train/eval the decoder model on the training/validation datasets
    # log the corresponding metrics and save the best model
    trainer.fit(model=lm, datamodule=dm, ckpt_path=ckpt_path, **fit_kwargs)

    # report some info about the training process
    log.info(f"LLR scaling factor after fit: {lm.llr_scaling.detach().item():.4f}")

    # evaluate the best model on the test set(s)
    # (note that this will issue a warning in a multi-gpu setting)
    trainer.test(model=lm, datamodule=dm)


if __name__ == "__main__":
    main()
