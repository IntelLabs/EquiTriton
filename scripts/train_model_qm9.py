from __future__ import annotations

from math import ceil
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import torch
from torch.optim.adamw import AdamW
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.data import Data as PyGGraph


class LightningQM9(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str = "./qm9_data",
        batch_size: int = 16,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        num_workers: int = 0,
    ):
        """
        Custom data module for QM9 dataset.

        Parameters
        ----------
        root_path : str, optional (default: "./qm9_data")
            Path to the QM9 dataset.
        batch_size : int, optional (default: 16)
            Number of samples in each mini-batch.
        train_frac : float, optional (default: 0.8)
            Fraction of data used for training.
        val_frac : float, optional (default: 0.1)
            Fraction of data used for validation.
        num_workers : int, optional (default: 0)
            Number of worker processes to use for loading data.

        Examples
        --------
        >>> dm = LightningQM9(root_path="/path/to/qm9_data", batch_size=32)

        Attributes
        ----------
        dataset : QM9
            Loaded QM9 dataset.
        hparams : dict
            Hyperparameters of the data module.

        Methods
        -------
        setup(stage: str)
            Setup data splits for training, validation and testing.
        train_dataloader()
            Returns a DataLoader instance for training data.
        val_dataloader()
            Returns a DataLoader instance for validation data.
        test_dataloader()
            Returns a DataLoader instance for testing data.
        """
        super().__init__()
        self.dataset = QM9(root_path)
        self.save_hyperparameters()

    def setup(self, stage: str):
        hparams = self.hparams
        num_samples = len(self.dataset)
        num_train = int(num_samples * hparams["train_frac"])
        num_val = int(num_samples * hparams["val_frac"])
        num_test = ceil(
            num_samples * (1 - (hparams["train_frac"] + hparams["val_frac"]))
        )
        # generate random splits
        train_split, val_split, test_split = random_split(
            self.dataset, lengths=[num_train, num_val, num_test]
        )
        self.splits = {"train": train_split, "val": val_split, "test": test_split}

    def train_dataloader(self):
        return DataLoader(
            self.splits["train"],
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.splits["val"],
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.splits["test"],
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )


class AtomWeightedMSE(nn.Module):
    """
    Calculates the mean-squared-error between predicted and targets,
    weighted by the number of atoms within each graph.

    From matsciml
    """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        atoms_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        if atoms_per_graph.size(0) != target.size(0):
            raise RuntimeError(
                "Dimensions for atom-weighted loss do not match:"
                f" expected atoms_per_graph to have {target.size(0)} elements; got {atoms_per_graph.size(0)}."
                "This loss is intended to be applied to scalar targets only."
            )
        # check to make sure we are broad casting correctly
        if (input.ndim != target.ndim) and target.size(-1) == 1:
            input.unsqueeze_(-1)
        # for N-d targets, we might want to keep unsqueezing
        while atoms_per_graph.ndim < target.ndim:
            atoms_per_graph.unsqueeze_(-1)
        # ensures that atoms_per_graph is type cast correctly
        squared_error = ((input - target) / atoms_per_graph.to(input.dtype)) ** 2.0
        return squared_error.mean()


class EquiTritonLitModule(pl.LightningModule):
    def __init__(
        self,
        model_class: type,
        model_kwargs,
        e_mean: float,
        e_std: float,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        atom_weighted_loss: bool = True,
    ):
        """
        Initializes the EquiTritonLitModule clas.

        Parameters
        ----------
        model_class : type
            Th class of the model to be used.
        model_kwargs : dict
            Keyword argument for the model initialization.
        e_mean : float
            The mean of the energy values.
        e_std : float
            The standard deviation of the energy values.
        lr : float, optional
            The learning rate (default is 1e-3) for AdamW.
        weight_decay : float, optional
            Weight decay value (default is 0.0).
        atom_weighted_loss : bool, optional
            Whether to use atom-weighted loss or not (default is True).
        """
        super().__init__()
        self.model = model_class(**model_kwargs)
        if atom_weighted_loss:
            self.loss = AtomWeightedMSE()
        else:
            self.loss = nn.MSELoss()
        self.output_head = nn.Linear(self.model.output_dim, 1)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def step(self, graph: PyGGraph, stage: Literal["train", "test", "val"]):
        """
        Performs a single step of the training, validation or testing
        process.

        Parameters
        ----------
        graph : PyGGraph
            The input graph.
        stage : Literal["train", "test", "val"]
            The current stage (training, testing or validation).

        Returns
        -------
        loss : float
            The calculated loss value.
        """
        g_z, z = self.model(graph)
        pred_energy = self.output_head(g_z)
        target_energy = graph.y[:, 12].unsqueeze(-1)
        norm_energy = (target_energy - self.hparams["e_mean"]) / self.hparams["e_std"]
        if self.hparams["atom_weighted_loss"]:
            loss = self.loss(pred_energy, norm_energy, torch.diff(graph.ptr))
        else:
            loss = self.loss(pred_energy, norm_energy)
        batch_size = getattr(graph, "batch_size", 1)
        self.log(
            f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=batch_size
        )
        return loss

    def training_step(self, batch):
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch):
        loss = self.step(batch, "val")
        return loss

    def test_step(self, batch):
        loss = self.step(batch, "test")
        return loss


if __name__ == "__main__":
    # use LightningCLI for easy configuration
    cli = LightningCLI(EquiTritonLitModule, LightningQM9)
