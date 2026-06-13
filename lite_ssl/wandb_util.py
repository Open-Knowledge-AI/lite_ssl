from pathlib import Path

import torch
import wandb
import lightning.pytorch as pl

from loguru import logger

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from lite_ssl.config import WANDB_ENTITY, WANDB_PROJECT


class SaveWeightsEveryEpochCallback(Callback):

    def __init__(self, save_dir: str | Path, filename: str):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.filename = filename

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self.save_dir is None:
            return

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / self.filename
            torch.save(pl_module.state_dict(), save_path)
        except Exception:
            try:
                fallback_path = Path(
                    f"/home/vaishp/ConKV/models/ConKV/{self.save_dir.name}/{self.filename}"
                )
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), fallback_path)
            except Exception:
                pass

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self.save_dir is None:
            return

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / self.filename
            torch.save(pl_module.state_dict(), save_path)
        except Exception:
            try:
                fallback_path = Path(
                    f"/home/vaishp/ConKV/models/ConKV/{self.save_dir.name}/{self.filename}"
                )
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), fallback_path)
            except Exception:
                pass


class SaveWeightsEveryNEpochCallback(Callback):

    def __init__(self, save_dir: str | Path | None, filename: str, save_every: int = 10):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.filename = filename
        self.save_every = save_every

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self.save_dir is None:
            return

        if (trainer.current_epoch + 1) % self.save_every != 0:
            return

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f"{self.filename}-{trainer.current_epoch}.pt"
            torch.save(pl_module.state_dict(), save_path)
        except Exception:
            try:
                fallback_path = Path(
                    f"/home/vaishp/ConKV/models/ConKV/{self.save_dir.name}/{self.filename}-{trainer.current_epoch}.pt"
                )
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), fallback_path)
            except Exception:
                pass

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self.save_dir is None:
            return

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f"last.ckpt"
            torch.save(pl_module.state_dict(), save_path)
        except Exception:
            try:
                fallback_path = Path(
                    f"/home/vaishp/ConKV/models/ConKV/{self.save_dir.name}/last.ckpt"
                )
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), fallback_path)
            except Exception:
                pass


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_version_counter = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if not trainer.is_global_zero:
            return

        try:
            trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)

            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

            if self.save_weights_only:
                self.rename_keys(filepath)

            wandb_run = wandb.run
            need_to_upload = (
                wandb_run is not None
                and not wandb_run.disabled
                and not wandb_run.offline
                and not wandb_run._is_finished
            )

            if need_to_upload:
                wandb.save(
                    filepath,
                    base_path=self.dirpath,
                    policy="now" if self.save_last else "live",
                )
                logger.info(f"Checkpoint saved to {filepath} and uploaded to WandB.")
            else:
                logger.info(f"Checkpoint saved to {filepath} but WandB is not active.")

        except Exception as e:
            logger.error(f"Failed to save checkpoint to {filepath}: {e}")

    def rename_keys(self, filepath: str):
        state_dict = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)
        # rename the keys to remove 'model.'
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        # save the state dict back to the file
        try:
            torch.save(state_dict, filepath)
            logger.info(f"Removed 'model.' prefix from state dict keys in {filepath}.")
        except Exception as e:
            logger.error(f"Failed to save the modified state dict back to {filepath}: {e}")

    def get_checkpoint_from_wandb(self, run_id: str):
        """
        Retrieve a checkpoint file from WandB using the run ID and filename.
        """
        if wandb.run is None or wandb.run.disabled:
            raise RuntimeError("WandB is not initialized or disabled.")

        run_path = f"{WANDB_PROJECT}/{run_id}"
        if WANDB_ENTITY is not None:
            run_path = f"{WANDB_ENTITY}/{run_path}"

        try:
            file = wandb.run.restore(
                name=f"{self.filename}{self.FILE_EXTENSION}",
                run_path=run_path,
                replace=True,
                root=self.dirpath,
            )
        except ValueError as e:
            logger.error(f"Error retrieving checkpoint from WandB: {e}")
            file = None

        if file is None:
            raise FileNotFoundError(
                f"Checkpoint {self.filename} not found in WandB for run {run_id}."
            )
        else:
            # get the path to the file
            path_to_file = file.name
            # close the file
            try:
                file.close()
            except Exception as e:
                logger.warning(f"Failed to close the file {file.name}: {e}")
            # return the path to the file
            return path_to_file


class TeacherCheckpoint(MyModelCheckpoint):

    def __init__(self, prefix="teacher_model", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    def rename_keys(self, filepath: str):
        was_inside_state_dict = False

        state_dict = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)
        if "state_dict" in state_dict:
            was_inside_state_dict = True
            state_dict = state_dict["state_dict"]

        # rename the keys to remove 'model.'
        state_dict = {
            k.replace(f"{self.prefix}.", "", 1): v
            for k, v in state_dict.items()
            if f"{self.prefix}" in k
        }
        # save the state dict back to the file
        if was_inside_state_dict:
            state_dict = {"state_dict": state_dict}

        try:
            torch.save(state_dict, filepath)
            logger.info(f"Removed '{self.prefix}.' prefix from state dict keys in {filepath}.")
        except Exception as e:
            logger.error(f"Failed to save the modified state dict back to {filepath}: {e}")
