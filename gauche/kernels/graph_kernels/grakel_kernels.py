from functools import lru_cache
from typing import List, Optional

import networkx as nx
import torch
from gpytorch import Module
from grakel import graph_from_networkx
from grakel.kernels import (
    EdgeHistogram,
    GraphletSampling,
    NeighborhoodHash,
    RandomWalk,
    RandomWalkLabeled,
    ShortestPath,
    VertexHistogram,
    WeisfeilerLehman,
)


class _GraphKernel(Module):
    """
    A base class suporting external graph kernels.
    The external kernel must have a method `fit_transform`, which, when
    evaluated on an `Inputs` instance `X`, returns a scaled kernel matrix
    v * k(X, X).

    As gradients are not propagated through to the external kernel, outputs are
    cached to avoid repeated computation.
    """

    def __init__(
        self,
        dtype=torch.float,
    ) -> None:
        super().__init__()
        self.node_label = None
        self.edge_label = None
        self._scale_variance = torch.nn.Parameter(
            torch.tensor([0.1], dtype=dtype)
        )

    def scale(self, S: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self._scale_variance) * S

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scale(self.kernel(X))

    def kernel(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")


class VertexHistogramKernel(_GraphKernel):
    """
    A GraKel wrapper for the vertex histogram kernel.
    This kernel requires node labels to be specified.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/vertex_histogram.html
    for more details.
    """

    def __init__(
        self,
        node_label: str,
        dtype=torch.float,
    ):
        super().__init__(dtype=dtype)
        self.node_label = node_label

    @lru_cache(maxsize=5)
    def kernel(self, X: List[nx.Graph], **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            VertexHistogram(**grakel_kwargs).fit_transform(X)
        ).float()


class EdgeHistogramKernel(_GraphKernel):
    """
    A GraKel wrapper for the edge histogram kernel.
    This kernel requires edge labels to be specified.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/edge_histogram.html
    for more details.
    """

    def __init__(self, edge_label, dtype=torch.float):
        super().__init__(dtype=dtype)
        self.edge_label = edge_label

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            EdgeHistogram(**grakel_kwargs).fit_transform(X)
        ).float()


class WeisfeilerLehmanKernel(_GraphKernel):
    """
    A GraKel wrapper for the Weisfeiler-Lehman kernel.
    This kernel needs node labels to be specified and
    can optionally use edge labels for the base kernel.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/weisfeiler_lehman.html
    for more details.
    """

    def __init__(
        self,
        node_label: str,
        edge_label: Optional[str] = None,
        dtype=torch.float,
    ):
        super().__init__(dtype=dtype)
        self.node_label = node_label
        self.edge_label = edge_label

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            WeisfeilerLehman(**grakel_kwargs).fit_transform(X)
        ).float()


class NeighborhoodHashKernel(_GraphKernel):
    """
    A GraKel wrapper for the neighborhood hash kernel.
    This kernel requires node labels to be specified.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/neighborhood_hash.html
    for more details.
    """

    def __init__(self, node_label: str, dtype=torch.float):
        super().__init__(dtype=dtype)
        self.node_label = node_label

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            NeighborhoodHash(**grakel_kwargs).fit_transform(X)
        ).float()


class RandomWalkKernel(_GraphKernel):
    """
    A GraKel wrapper for the random walk kernel.
    This kernel only works on unlabelled graphs.
    See RandomWalkLabeledKernel for labelled graphs.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/random_walk.html
    for more details.
    """

    def __init__(self, dtype=torch.float):
        super().__init__(dtype=dtype)

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            RandomWalk(**grakel_kwargs).fit_transform(X)
        ).float()


class RandomWalkLabeledKernel(_GraphKernel):
    """
    A GraKel wrapper for the random walk kernel.
    This kernel requires node labels to be specified.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/random_walk.html
    for more details.
    """

    def __init__(self, node_label: str, dtype=torch.float):
        super().__init__(dtype=dtype)
        self.node_label = node_label

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            RandomWalkLabeled(**grakel_kwargs).fit_transform(X)
        ).float()


class ShortestPathKernel(_GraphKernel):
    """
    A GraKel wrapper for the shortest path kernel.
    This kernel only works on unlabelled graphs.
    See ShortestPathLabeledKernel for labelled graphs.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/shortest_path.html
    for more details.
    """

    def __init__(self, dtype=torch.float):
        super().__init__(dtype=dtype)

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            ShortestPath(**grakel_kwargs, with_labels=False).fit_transform(X)
        ).float()


class ShortestPathLabeledKernel(_GraphKernel):
    """
    A GraKel wrapper for the shortest path kernel.
    This kernel requires node labels to be specified.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/shortest_path.html
    for more details.
    """

    def __init__(self, node_label: str, dtype=torch.float):
        super().__init__(dtype=dtype)
        self.node_label = node_label

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            ShortestPath(**grakel_kwargs, with_labels=True).fit_transform(X)
        ).float()


class GraphletSamplingKernel(_GraphKernel):
    """
    A GraKel wrapper for the graphlet sampling kernel.
    This kernel only works on unlabelled graphs.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/graphlet_sampling.html
    for more details.
    """

    def __init__(self, dtype=torch.float):
        super().__init__(dtype=dtype)

    @lru_cache(maxsize=5)
    def kernel(self, X: torch.Tensor, **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        X = graph_from_networkx(
            X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        )

        return torch.tensor(
            GraphletSampling(**grakel_kwargs).fit_transform(X)
        ).float()
