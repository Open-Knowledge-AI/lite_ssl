import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,GRAPH"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import json
import argparse

from pprint import pprint

import torch
import lightning.pytorch as pl

from loguru import logger
from ml_collections import ConfigDict
from lightning.pytorch.loggers import WandbLogger

from lite_ssl.data import make_loaders
from lite_ssl.config import MODELS_DIR, WANDB_PROJECT
from lite_ssl.wandb_util import SaveWeightsEveryNEpochCallback
from lite_ssl.trainer import (
    LeJepaSSTrainer,
    LeJepaMSTrainer,
    DiNoTrainer,
    DiNoBOTTrainer,
)

torch.set_float32_matmul_precision("medium")


def is_rank_zero():
    """
    Robust check for rank 0 across common launchers:
      - torchrun / torch.distributed: LOCAL_RANK (local) or RANK (global)
      - SLURM: SLURM_PROCID (global)
    If none set, default to True (single-process).
    """
    # global rank
    rank_env = os.environ.get("RANK", None)
    if rank_env is None:
        rank_env = os.environ.get("SLURM_PROCID", None)
    # fallback to local rank
    local_rank_env = os.environ.get("LOCAL_RANK", None)

    try:
        if rank_env is not None:
            return int(rank_env) == 0
        if local_rank_env is not None:
            return int(local_rank_env) == 0
    except Exception:
        pass
    # default when no env var: single-process -> treat as rank 0
    return True


def set_run_name(cfg):
    run_name = f"{cfg.dataset.name}"

    if hasattr(cfg, "subset") and cfg.subset.pct > 0:
        run_name += f"_sub={cfg.subset.pct / 100:.2f}"

    if hasattr(cfg.dataset, "con") and cfg.dataset.con.enable:
        run_name += f"-{cfg.dataset.con.type}"

    if hasattr(cfg.loss, "id"):
        run_name += f"-{cfg.loss.id}"

    run_name += f"-{cfg.model.type}"

    logger.info(f"Run name: {run_name}")
    cfg.run_name = run_name
    return cfg


def train(cfg, fast_dev_run=False):
    """
    Rank-aware WandB initialization and checkpoint handling.
    Only rank 0 will create the wandb logger and touch WandB API.
    Non-zero ranks will skip wandb and not attempt to read experiment.id.
    """

    rank0 = is_rank_zero()
    if hasattr(cfg, "run_id") and cfg.run_id is not None and rank0:
        logger.info(f"Using existing run ID: {cfg.run_id}")

    # Initialize wandb only on rank 0
    wandb_logger = None
    if rank0:
        # create the WandbLogger in rank 0 only
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT, name=cfg.run_name, id=cfg.run_id, allow_val_change=True
        )

        # sanity check run_id / wandb id match (only on rank 0)
        if (
            hasattr(cfg, "run_id")
            and cfg.run_id is not None
            and cfg.run_id != wandb_logger.experiment.id
        ):
            logger.error(
                f"Given run ID {cfg.run_id} does not match the WandB run ID {wandb_logger.experiment.id}."
            )
            raise ValueError("Run ID mismatch. Please check your configuration.")

    # Create checkpoints_dir only on rank 0 (other ranks don't need to create it)
    if rank0:
        try:
            checkpoints_dir = (
                MODELS_DIR / wandb_logger.experiment.project_name() / wandb_logger.experiment.id
            )
        except Exception:
            # fallback if project_name not available
            try:
                checkpoints_dir = MODELS_DIR / wandb_logger.experiment.id
            except Exception:
                checkpoints_dir = MODELS_DIR / (cfg.run_name or "run")
        try:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create checkpoints directory {checkpoints_dir}: {e}")
        logger.info(f"[rank0] Checkpoints directory: {checkpoints_dir}")
    else:
        # non-zero ranks: set a local placeholder path (not used for saving)
        checkpoints_dir = MODELS_DIR / (cfg.run_name or "run") / "rank_nonzero"
        # do not attempt to create it to avoid races

    # Build callbacks. MyModelCheckpoint may attempt WandB interactions; only initialize on rank 0.
    save_weights_n_epoch = SaveWeightsEveryNEpochCallback(
        save_dir=checkpoints_dir if rank0 else None, filename="epoch", save_every=10
    )

    # Build trainer: pass wandb_logger on rank 0, None on others.
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            save_weights_n_epoch,
        ],
        **cfg.pl.trainer.to_dict(),
        fast_dev_run=fast_dev_run,
    )

    train_loader, val_loader, norm, val_transform, ptrain_loader, pval_loader = make_loaders(cfg)

    if cfg.trainer == "lj_sc":
        module_cls = LeJepaSSTrainer
    elif cfg.trainer == "lj_ms":
        module_cls = LeJepaMSTrainer
    elif cfg.trainer == "dino":
        module_cls = DiNoTrainer
    elif cfg.trainer == "dinobot":
        module_cls = DiNoBOTTrainer
    else:
        raise ModuleNotFoundError(f"No module named {cfg.trainer} found!")

    # The following WandB-dependent operations (downloading checkpoints, using wandb.experiment.id)
    # must be run only on rank 0 to avoid None/absent experiment IDs.
    if rank0 and hasattr(cfg, "run_id") and cfg.run_id is not None:
        # only rank 0 attempts to restore from WandB
        if not (checkpoints_dir / "last.ckpt").exists():
            try:
                save_weights_n_epoch.get_checkpoint_from_wandb(
                    run_id=wandb_logger.experiment.id,
                )
                logger.success(
                    f"Checkpoint {save_weights_n_epoch.filename} restored from WandB for run {cfg.run_id}."
                )
            except FileNotFoundError as e:
                logger.error(str(e))

    # Set cfg.run_id from wandb only on rank 0
    if rank0 and wandb_logger is not None:
        cfg.run_id = wandb_logger.experiment.id
    # Non-zero ranks keep cfg.run_id as-is (possibly None). That's fine; they don't access wandb.

    module = module_cls(
        config=cfg, normalisation=norm, probe_valid_dl=pval_loader, probe_train_dl=ptrain_loader
    )

    candidate_last = checkpoints_dir / "last.ckpt"
    # Trainer.fit will be run on all ranks. Checkpoint loading / saving is rank-aware.
    trainer.fit(
        module,
        train_loader,
        val_loader,
        ckpt_path=candidate_last if candidate_last.exists() else None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast dev run",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Convert the config to a ConfigDict
    cfg = ConfigDict(cfg, convert_dict=True)
    cfg = set_run_name(cfg)

    if is_rank_zero():
        pprint(cfg.to_dict())

    train(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
