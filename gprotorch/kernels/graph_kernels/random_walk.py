"""
Contains GPyTorch random walk kernels on undirected graphs.
Notes: requires pytorch 1.8.0+ for torch.kron()
"""

import torch
from gpytorch.kernels import Kernel
from gpytorch.kernels.kernel import default_postprocess_script
from math import factorial
from gpytorch.constraints import Positive
from .graph_kernel_utils import normalise_covariance


class RandomWalkUnlabelled(Kernel):
    """
    Calculates the random walk kernel introduced by Gärtner et al LTKM, 129-143, 2003
    on unlabelled graphs using the spectral decomposition method introduced in
    Vishwanathan et al JLMR, 11:1201–1242, 2010.
    """

    def __init__(self,
                 normalise=True,
                 uniform_probabilities=True,
                 p=None,
                 series_type='geometric',
                 weight_constraint=None,
                 weight_prior=None,
                 **kwargs):
        """
        Initialises the kernel class.

        Args:
            normalise: whether to normalise the graph kernels
            uniform_probabilities: whether the starting and stopping probabilities are uniform across nodes
            (as in Vishwanathan et al) or whether they are all set to 1 (as in Gärtner et al)
            (both are equivalent if normalisation is performed)
            p: whether to truncate the power series and only calculate p entries (int)
            or calculate the entire power series (None)
            series_type: whether to use a geometric or exponential power series
            weight_constraint: the constraint to put on the weight parameter,
            Positive() by default, but in some circumstances Interval might be better()
            weight_prior: prior of the weight parameter
            **kwargs: base kernel class arguments
        """

        super(RandomWalkUnlabelled, self).__init__(**kwargs)

        self.normalise = normalise
        self.uniform_probabilities = uniform_probabilities
        self.p = p

        if series_type == 'geometric':
            self.geometric = True
        elif series_type == 'exponential':
            self.geometric = False
        else:
            raise NotImplementedError('The unlabelled random walk kernel only supports the'
                                      'seriess_type parameters "geometric" and "exponential"')

        self.register_parameter(
            name='raw_weight', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if weight_constraint is None:
            weight_constraint = Positive()

        self.register_constraint('raw_weight', weight_constraint)

        if weight_prior is not None:
            self.register_prior(
                'weight_prior',
                weight_prior,
                lambda m: m.weight,
                lambda m, v: m._set_weight(v)
            )

    @property
    def weight(self) -> torch.Tensor:
        return self.raw_weight_constraint.transform(self.raw_weight)

    @weight.setter
    def weight(self,
               value: torch.Tensor
               ) -> None:
        self._set_weight(value)

    def _set_weight(self,
                    value: torch.Tensor
                    ) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(raw_weight=self.raw_weight_constraint.inverse_transform(value))

    def forward(self,
                x1: list,
                x2: list,
                **params) -> torch.Tensor:
        """
        Calculates the covariance matrix between x1 and x2.

        Args:
            x1: list of (potentially edge-weighted) adjacency matrices
            x2: list of (potentially edge-weighted) adjacency matrices
            **params:

        Returns: covariance matrix

        """

        # check if the matrices are identical (order-dependent)
        x1_equals_x2 = (x1 == x2)

        # Step 1: perform spectral decomposition of all adjacency matrices

        eigen_vals_1, eigen_vecs_1 = [], []
        eigen_vals_2, eigen_vecs_2 = [], []

        for adj_mat in x1:
            val, vec = torch.symeig(adj_mat, eigenvectors=True)
            eigen_vals_1.append(val)
            eigen_vecs_1.append(vec)

        if x1_equals_x2:
            eigen_vals_2 = eigen_vals_1
            eigen_vecs_2 = eigen_vecs_1

        else:
            for adj_mat in x2:
                val, vec = torch.symeig(adj_mat, eigenvectors=True)
                eigen_vals_2.append(val)
                eigen_vecs_2.append(vec)

        # Step 2: calculate all flanking factors

        flanking_factors_1 = []
        flanking_factors_2 = []

        for eigen_vec in eigen_vecs_1:

            start_stop_probs = torch.unsqueeze(torch.ones(eigen_vec.size(0)), 0)
            if self.uniform_probabilities:
                start_stop_probs.div_(eigen_vec.size(0))

            flanking_factors_1.append(
                torch.matmul(start_stop_probs, eigen_vec)
            )

        if x1_equals_x2:
            flanking_factors_2 = flanking_factors_1

        else:
            for eigen_vec in eigen_vecs_2:

                start_stop_probs = torch.unsqueeze(torch.ones(eigen_vec.size(0)), 0)
                if self.uniform_probabilities:
                    start_stop_probs.div_(eigen_vec.size(0))

                flanking_factors_2.append(
                    torch.matmul(start_stop_probs, eigen_vec)
                )

        # Step 3: populate the covariance matrix

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in range(len(x1)):
            for idx2 in range(len(x2)):

                # if applicable, fill lower triangular part
                # with already calculated upper triangular part
                if x1_equals_x2 and idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]

                else:

                    flanking_factor = torch.kron(flanking_factors_1[idx1], flanking_factors_2[idx2])
                    diagonal = self.weight * torch.kron(eigen_vals_1[idx1], eigen_vals_2[idx2])

                    # if only random walks of a set length are considered
                    # evaluate power series iteratively
                    if self.p is not None:
                        power_series = torch.zeros_like(diagonal)
                        temp_diagonal = torch.ones_like(diagonal)

                        for k in range(self.p):
                            power_series.add_(temp_diagonal)
                            temp_diagonal.mul_(diagonal)
                            if not self.geometric:
                                temp_diagonal.div_(factorial(k))

                        power_series = torch.diagflat(power_series)

                    # otherwise use closed-from expressions
                    else:
                        if self.geometric:
                            power_series = torch.diagflat(1 / (1 - diagonal))
                        else:
                            power_series = torch.diagflat(torch.exp(diagonal))

                    covar_mat[idx1, idx2] = torch.matmul(
                        flanking_factor,
                        torch.matmul(
                            power_series,
                            torch.transpose(flanking_factor, -2, -1)
                        )
                    )

        if self.normalise:
            covar_mat = normalise_covariance(covar_mat)

        return covar_mat
