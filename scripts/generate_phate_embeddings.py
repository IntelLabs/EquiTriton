from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

import torch
from rdkit import Chem
import wandb
import numpy as np
from tqdm import tqdm
from phate import PHATE
from e3nn import o3
import pytorch_lightning as pl

from equitriton.model.lightning import EquiTritonLitModule, LightningQM9
from equitriton.utils import separate_embedding_irreps


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


def run_phate_projection(
    results: list[dict], irreps: o3.Irreps, **phate_kwargs
) -> dict[str, np.ndarray]:
    phate_kwargs.setdefault("knn", 10)
    phate_kwargs.setdefault("random_state", 21516)
    embeddings = torch.vstack([r["embeddings"][1] for r in results]).numpy()
    # separate embeddings into individual chunks
    chunk_dict = separate_embedding_irreps(embeddings, irreps, return_numpy=True)
    embeddings_dict = {}
    for order, chunk in chunk_dict.items():
        print(f"Running PHATE on order {order}")
        # collect up all the embeddings
        phate = PHATE(**phate_kwargs)
        phate_embeddings = phate.fit_transform(chunk)
        embeddings_dict[order] = phate_embeddings
    # run once more on the full embedding set
    phate = PHATE(**phate_kwargs)
    phate_embeddings = phate.fit_transform(embeddings)
    embeddings_dict["full"] = phate_embeddings
    return embeddings_dict


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "artifact_path", type=str, help="wandb path to a model artifact."
    )
    pl.seed_everything(215162)

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
    model = EquiTritonLitModule.load_from_checkpoint(str(ckpt_path)).eval()

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    results = []
    all_smi = []
    all_error = []
    score_dict = {}
    for index, batch in tqdm(
        enumerate(test_loader),
        desc="Batches to process",
        leave=False,
        position=1,
        total=len(test_loader),
    ):
        embeddings = model.model.embed(batch.to("cuda"))
        with torch.no_grad():
            g_z, z = model.model(batch)
            pred_energies = model.output_head(g_z)
            # un-normalize energy
            pred_energies = (model.hparams["e_std"] * pred_energies) + model.hparams[
                "e_mean"
            ]
            # retrieve targets
            target_energies = batch.y[:, 12].unsqueeze(-1)
            error = (pred_energies - target_energies).pow(2.0).cpu().tolist()
        mols = graph_to_rdkit(batch)
        scores = calculate_scores_for_batch(mols)
        package = {
            "embeddings": embeddings["graph_z"],
            "scores": scores,
            "smi": batch.smiles,
            "error": error,
        }
        all_smi.extend(batch.smiles)
        all_error.extend(error)
        # reformat scores into a flat dictionary
        for score in scores:
            for key, value in score.items():
                if key not in score_dict:
                    score_dict[key] = []
                score_dict[key].append(value)
        results.append(package)
    print("Running PHATE on each Irreps")
    phate_embeddings = run_phate_projection(
        results, model.model.initial_layer.output_irreps
    )
    to_save = {"phate": phate_embeddings, "data": results}
    # save a local version of the results
    torch.save(to_save, Path(artifact_dir).joinpath("results.pt"))
    # formatting stuff to log to wandb
    embedding_table = wandb.Table(
        columns=["F1", "F2"], data=phate_embeddings["full"].tolist()
    )
    for key, array in phate_embeddings.items():
        if key != "full":
            for axis in [0, 1]:
                embedding_table.add_column(name=f"O{key}_{axis}", data=array[:, axis])
    # this initializes the table
    joint_table = wandb.Table(columns=["smiles"])
    # i'm not sure why, but the table kept fussing about not being
    # to add the list of smiles directly, which is why it's written as a loop
    for smi in all_smi:
        joint_table.add_data(smi)
    joint_table.add_column(name="squared_error_eV", data=all_error)
    # now add the descriptors as well
    for key, value in score_dict.items():
        joint_table.add_column(name=key, data=value)
    # package stuff up and log to wandb
    inference_artifact = wandb.Artifact("qm9_inference", type="eval")
    inference_artifact.add(embedding_table, "phate")
    inference_artifact.add(joint_table, "descriptors")
    inference_run.log_artifact(inference_artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
