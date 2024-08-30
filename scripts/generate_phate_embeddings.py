from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

import torch
from rdkit import Chem
import wandb
import numpy as np
from tqdm import tqdm
from phate import PHATE

from equitriton.model.lightning import EquiTritonLitModule, LightningQM9


def graph_to_rdkit(batched_graph):
    mols = [Chem.MolFromSmiles(smi, sanitize=False) for smi in batched_graph.smiles]
    return mols


def score_molecule(molecule) -> dict[str, int]:
    enum = {"SP": 1, "SP2": 2, "SP3": 3}
    scores = {"stereo": 0, "hybrid": 0, "aromatic": 0, "heavy_atoms": 0}
    for atom in tqdm(
        molecule.GetAtoms(), desc="Atoms in a molecule", leave=False, position=3
    ):
        hybrid = enum.get(str(atom.GetHybridization()), 0)
        # loop over bonds on the atom to check if it has stereoisomers
        has_stereo = any(
            [
                True if b.GetStereo() == Chem.BondStereo.STEREOE else False
                for b in atom.GetBonds()
            ]
        )
        s = 2 if has_stereo else 1
        r = int(atom.GetIsAromatic())
        heavy_atoms = sum(
            [neighbor.GetAtomicNum() > 1 for neighbor in atom.GetNeighbors()]
        )
        scores["stereo"] += s
        scores["hybrid"] += hybrid
        scores["aromatic"] += r
        scores["heavy_atoms"] += heavy_atoms
    return scores


def calculate_scores_for_batch(molecules) -> list[dict[str, int]]:
    """
    Calculates scores for every graph in a batch.
    """
    scores = [
        score_molecule(mol)
        for mol in tqdm(molecules, desc="Scoring molecules", leave=False, position=2)
    ]
    return scores


def run_phate_projection(results: list[dict], **phate_kwargs) -> np.ndarray:
    phate_kwargs.setdefault("knn", 10)
    phate_kwargs.setdefault("random_state", 21516)
    # collect up all the embeddings
    embeddings = torch.vstack([r["embeddings"][1] for r in results]).numpy()
    phate = PHATE(**phate_kwargs)
    phate_embeddings = phate.fit_transform(embeddings)
    return phate_embeddings


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "artifact_path", type=str, help="wandb path to a model artifact."
    )

    args = parser.parse_args()

    inference_run = wandb.init(
        job_type="eval",
        entity="laserkelvin",
        project="equitriton-qm9",
        tags=["inference", "embeddings", "qm9"],
    )

    artifact = inference_run.use_artifact(args.artifact_path, type="model")
    artifact_dir = artifact.download()
    ckpt_path = Path(artifact_dir).joinpath("model.ckpt")

    datamodule = LightningQM9("./qm9_data", num_workers=0)
    model = EquiTritonLitModule.load_from_checkpoint(str(ckpt_path))

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    results = []
    for index, batch in tqdm(
        enumerate(test_loader), desc="Batches to process", leave=False, position=1
    ):
        embeddings = model.model.embed(batch.to("cuda"))
        mols = graph_to_rdkit(batch)
        scores = calculate_scores_for_batch(mols)
        package = {
            "embeddings": embeddings["graph_z"],
            "scores": scores,
            "smi": batch.smiles,
        }
        results.append(package)
    phate_embeddings = run_phate_projection(results)
    to_save = {"phate": phate_embeddings, "data": results}
    torch.save(to_save, Path(artifact_dir).joinpath("results.pt"))


if __name__ == "__main__":
    main()
