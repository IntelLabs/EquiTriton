from __future__ import annotations

from typing import Literal, Any, Callable

import torch
from torch import nn
import e3nn
from e3nn import o3
from torch_scatter import scatter
from matplotlib import pyplot as plt
from torch_geometric.data import Data as PyGGraph

from equitriton.utils import spherical_harmonics_irreps
from equitriton.sph_harm.direct import triton_spherical_harmonic


class AtomEmbedding(nn.Module):
    def __init__(self, num_atoms: int, atom_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atoms, atom_dim, padding_idx=0)

    def forward(self, atomic_numbers: torch.LongTensor) -> torch.Tensor:
        return self.embedding(atomic_numbers)


class EdgeEmbedding(nn.Module):
    def __init__(self, num_basis: int, radius_cutoff: float = 6.0, **kwargs):
        """
        This module embeds edges in a graph with an EdgeEmbedding object.

        Parameters
        ----------
        num_basis : int, optional
            The number of basis functions. Defaults to 1.
        radius_cutoff : float, optional
            The maximum radius up to which basis functions are defined. Defaults to 6.0.

        Optional kwargs
        ---------------
        basis : str, optional
            The type of basis function to use. Defaults to 'bessel'.
        start : float, optional
            The starting point in the distance grid used in the radial basis.
        cutoff : bool, optional
            Whether or not to apply a cutoff to the basis functions.

        Returns
        -------
        torch.Tensor
            A tensor representing the embedding of edges with shape (num_edges, num_basis).

        Examples
        --------
        >>> # Define an instance of EdgeEmbedding with 4 basis functions and a radius cutoff of 10.
        >>> embedder = EdgeEmbedding(num_basis=4, radius_cutoff=10.0)
        """
        super().__init__()
        kwargs.setdefault("basis", "bessel")
        kwargs.setdefault("start", 0.0)
        kwargs.setdefault("cutoff", True)
        self.num_basis = num_basis
        self.radius_cutoff = radius_cutoff
        self.basis_kwargs = kwargs

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        basis_funcs = e3nn.math.soft_one_hot_linspace(
            distances,
            number=self.num_basis,
            end=self.radius_cutoff,
            **self.basis_kwargs,
        )
        return basis_funcs * self.num_basis**0.5


class SphericalHarmonicEmbedding(nn.Module):
    def __init__(
        self,
        l_values: list[int],
        normalize: bool = True,
        normalization: Literal["norm", "integral", "component"] = "integral",
        use_e3nn: bool = False,
    ):
        """
        Projects cartesian coordinates onto a specified spherical harmonic basis.

        Arguments mainly implement an equivalent interface to ``e3nn``,
        up to just directly using the ``e3nn`` spherical harmonics
        implementation.

        Parameters
        ----------
        l_values : list[int]
            List of l values of spherical harmonics to use as a basis.
        normalize : bool, optional
            Whether to normalize coordinates before passing into the
            embedding step.
        normalization : Literal["norm", "integral", "component"], optional
            Normalization scheme to use for the embeddings. By default
            uses ``integral``, which is the only method implemented for
            the Triton kernels.
        use_e3nn : bool, optional
            Whether to directly use ``e3nn`` spherical harmonics,
            by default False.
        """
        super().__init__()
        self.l_values = list(sorted(l_values))
        self.irreps = spherical_harmonics_irreps(self.l_values, num_feat=1)
        self.normalize = normalize
        self.normalization = normalization
        self.use_e3nn = use_e3nn

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.use_e3nn:
            if self.normalize:
                coords = torch.nn.functional.normalize(coords, dim=-1)
            # TODO concatenation is slow; work directly on pre-allocated tensors
            outputs = [triton_spherical_harmonic(l, coords) for l in self.l_values]
            outputs = torch.cat(outputs, dim=-1)
            if self.normalization == "integral":
                outputs /= (4.0 * torch.pi) ** 0.5
            return outputs
        else:
            return o3.spherical_harmonics(
                self.irreps, coords, self.normalize, self.normalization
            )


class InteractionBlock(nn.Module):
    def __init__(
        self,
        atomic_dim: int | o3.Irreps,
        l_values: list[int],
        edge_dim: int,
        hidden_dim: int,
        radius_cutoff: float,
        degree_norm: float,
        edge_kwargs: dict[str, Any] = {},
        sph_harm_kwargs: dict[str, Any] = {},
        activation: Callable = nn.functional.silu,
    ):
        """
        A module that combines radial basis with spherical harmonics to
        describe molecular interactions.

        Parameters
        ----------
        atomic_dim : int | o3.Irreps
            Dimension of the atomic features. If int, it is treated as a
            single irreducible representation.
        l_values : list[int]
            Values of the spherical harmonic order. If the Triton harmonics
            are being used, this does not need to be contiguous.
        edge_dim : int
            Dimension of the edge features.
        hidden_dim : int
            Hidden dimension for the fully connected network.
        radius_cutoff : float
            Cutoff radius for the radial basis.
        degree_norm : float
            Normalization factor for the degree of the graph.
        edge_kwargs : dict[str, Any], optional
            Keyword arguments for the EdgeEmbedding module. Defaults to {}.
        sph_harm_kwargs : dict[str, Any], optional
            Keyword arguments for the SphericalHarmonicEmbedding module.
            Defaults to {}.
        activation : Callable, optional
            Activation function for the fully connected network. Defaults to
            nn.functional.silu.

        Notes
        -----
        The `degree_norm` attribute is set as a property and effectively
        represents the average number of neighbors in other models.

        Examples
        --------
        >>> block = InteractionBlock(atomic_dim=8, l_values=[0, 1],
            edge_dim=16, hidden_dim=32)
        >>> block.sph_irreps
        ['1x0e', '2x0e']
        """
        sph_harm_kwargs.setdefault("use_e3nn", False)

        super().__init__()
        # this is effectively the average number of neighbors in other models
        self.degree_norm = degree_norm
        # treat atom features as invariant
        if isinstance(atomic_dim, int):
            atomic_irreps = f"{atomic_dim}x0e"
        else:
            atomic_irreps = atomic_dim
        self.atomic_irreps = atomic_irreps
        self.l_values = list(sorted(l_values))
        # these two attributes are similar but different: the former is used for describing
        # the basis itself, and the latter is for actually specifying the weights
        self.sph_irreps = spherical_harmonics_irreps(self.l_values, num_feat=1)
        self.output_irreps = spherical_harmonics_irreps(
            self.l_values, num_feat=hidden_dim
        )
        # tensor product is the final bit the combines the radial basis with the spherical
        # harmonics
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.atomic_irreps,
            self.sph_irreps,
            self.output_irreps,
            shared_weights=False,
        )
        self.edge_basis = EdgeEmbedding(edge_dim, radius_cutoff, **edge_kwargs)
        self.spherical_harmonics = SphericalHarmonicEmbedding(
            l_values, **sph_harm_kwargs
        )
        self.fc = e3nn.nn.FullyConnectedNet(
            [edge_dim, hidden_dim, self.tensor_product.weight_numel], activation
        )

    @property
    def num_projections(self) -> int:
        """Returns the expected number of projections."""
        return sum([2 * l + 1 for l in self.l_values])

    @property
    def output_dim(self) -> int:
        """Returns the dimensionality of the output."""
        return self.output_irreps.dim

    def forward(
        self,
        atomic_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        High-level description:

        1. Project cartesian coordinates onto spherical harmonic basis
        2. Project interatomic distances onto radial (bessel) basis
        3. Transform radial basis functions with learnable weights
        4. Compute tensor product between scalar atom features and spherical harmonic basis
        5. Update node features
        """
        edge_dist = coords[edge_index[0]] - coords[edge_index[1]]
        sph_harm = self.spherical_harmonics(edge_dist)
        # calculate atomic distances, embed, and transform them
        edge_basis = self.edge_basis(edge_dist.norm(dim=-1))
        edge_z = self.fc(edge_basis)
        # compute tensor product
        messages = self.tensor_product(atomic_features[edge_index[0]], sph_harm, edge_z)
        # update node features
        hidden_feats = (
            scatter(messages, edge_index[1], dim=0, dim_size=atomic_features.size(0))
            / self.degree_norm
        )
        return hidden_feats


class ScalarReadoutLayer(nn.Module):
    def __init__(self, hidden_irreps: o3.Irreps, output_dim: int):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.output_irreps = o3.Irreps(f"{output_dim}x0e")
        self.output_layer = o3.Linear(
            irreps_in=hidden_irreps, irreps_out=self.output_irreps
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        return self.output_layer(node_feats)


class EquiTritonModel(nn.Module):
    def __init__(
        self,
        initial_atom_dim: int,
        num_layers: int,
        output_dim: int,
        l_values: list[int],
        edge_dim: int,
        hidden_dim: int,
        radius_cutoff: float,
        degree_norm: float,
        edge_kwargs: dict[str, Any] = {},
        sph_harm_kwargs: dict[str, Any] = {},
        activation: Callable = nn.functional.silu,
        num_atoms: int = 100,
        skip_connections: bool = True,
    ):
        """
        A neural network model designed for processing molecular graphs.

        This class implements a hierarchical architecture with multiple interaction blocks,
        allowing for efficient and scalable processing of large molecular datasets.

        Parameters
        =============
            initial_atom_dim : int
                The dimensionality of the atomic embeddings.
            num_layers : int
                The number of convolutional layers to use.
            output_dim : int
                The dimensionality of the final scalar features.
            l_values : list[int]
                A list of spherical harmonics order to consider. If using the Triton kernels,
                does not need to be contiguous.
            edge_dim : int
                The dimensionality of the edge features.
            hidden_dim : int
                The dimensionality of the hidden state in each interaction block.
            radius_cutoff : float
                The cutoff distance for radial basis functions.
            degree_norm : float
                The normalization constant for edge features. Typically square root of the average degree.
            edge_kwargs : dict[str, Any], optional
                Keyword arguments to pass to the InteractionBlock.
            sph_harm_kwargs : dict[str, Any], optional
                Keyword arguments to pass to the InteractionBlock. By default,
                the ``use_e3nn`` kwarg is set to False, which uses the Triton kernels instead.
            activation : Callable, optional
                The activation function to use in each interaction block. Defaults to nn.functional.silu.
            num_atoms : int, optional
                The number of atoms in the embedding table (i.e. unique elements). Defaults to 100.
            skip_connections : bool, optional
                Whether to enable residual connections between layers. Defaults to True.

        Examples
        ============
            >>> model = EquiTritonModel(...)
            >>> graph = PyGGraph(...).to(device="cuda")
            >>> graph_z, z = model(graph)
        """
        sph_harm_kwargs.setdefault("use_e3nn", False)

        super().__init__()
        self.atomic_embedding = AtomEmbedding(num_atoms, initial_atom_dim)
        self.initial_layer = InteractionBlock(
            initial_atom_dim,
            l_values,
            edge_dim,
            hidden_dim,
            radius_cutoff,
            degree_norm,
            edge_kwargs,
            sph_harm_kwargs,
            activation,
        )
        self.conv_layers = nn.ModuleDict()
        for layer_index in range(num_layers + 1):
            self.conv_layers[f"conv_{layer_index}"] = InteractionBlock(
                self.initial_layer.output_irreps,  # subsequent layers use irreps instead
                l_values,
                edge_dim,
                hidden_dim,
                radius_cutoff,
                degree_norm,
                edge_kwargs,
                sph_harm_kwargs,
                activation,
            )
        self.scalar_readout = ScalarReadoutLayer(
            self.initial_layer.output_irreps, output_dim
        )
        self.skip_connections = skip_connections
        self.output_dim = output_dim

    def visualize(self, **kwargs):
        """
        Visualize the sequence of tensor products within the model.

        Essentially, all this does is wrap around the ``tensor_product.visualize()``
        functionality, but also tacks on titles for each axis.
        """
        num_plots = len(self.conv_layers) + 1
        fig, axarray = plt.subplots(num_plots, 1, figsize=(3, 12))
        # make indexing easier
        axarray = axarray.flatten()

        self.initial_layer.tensor_product.visualize(ax=axarray[0], **kwargs)
        axarray[0].set_title("Input layer", loc="right")
        index = 1
        for layer_name, layer in self.conv_layers.items():
            ax = axarray[index]
            layer.tensor_product.visualize(ax=ax, **kwargs)
            ax.set_title(layer_name, loc="right")
            index += 1
        fig.tight_layout()
        return fig, axarray

    def forward(self, graph: PyGGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a generic equivariant convolution model.

        Parameters
        ----------
        graph : PyGGraph
            PyG graph structure, which may be batched or a single graph.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            2-tuple of outputs; first element are graph level
            outputs (from summing over nodes), second element
            is the node-level outputs.
        """
        # determine if the graph is batched or not
        is_batched = hasattr(graph, "ptr")
        for key in ["pos", "edge_index", "z"]:
            assert hasattr(graph, key)
        # get atom embeddings
        atom_z = self.atomic_embedding(graph.z)  # [nodes, initial_atom_dim]
        # first message passing step
        z = self.initial_layer(atom_z, graph.pos, graph.edge_index)
        outputs = {}
        for layer_name, layer in self.conv_layers.items():
            new_z = layer(z, graph.pos, graph.edge_index)
            # add residual connections
            if self.skip_connections and new_z.shape == z.shape:
                new_z += z
            z = new_z
            outputs[layer_name] = z
        # map final output as scalars
        z = self.scalar_readout(z)
        # latest node features are in z; we generate graph-level scalar features
        # by doing a scatter add
        if is_batched:
            graph_z = scatter(z, graph.batch, dim=0, dim_size=graph.batch_size)
        else:
            # for a single graph, just sum up the node features
            graph_z = z.sum(dim=0, keepdims=True)
        return graph_z, z
