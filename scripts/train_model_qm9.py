from __future__ import annotations

from pytorch_lightning.cli import LightningCLI
import torch

from equitriton.model.lightning import EquiTritonLitModule, LightningQM9


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # use LightningCLI for easy configuration
    cli = LightningCLI(
        EquiTritonLitModule, LightningQM9, save_config_kwargs={"overwrite": True}
    )
