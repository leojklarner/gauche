# GPU-compatible gpytorch version of the substring kernel implemented in https://github.com/henrymoss/BOSS/tree/master/boss/code/kernels/string
# algorithmic description in 
# [1] Beck, D., & Cohn, T. Learning kernels over strings using Gaussian processes. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 67-73).
# or appendix of [2] Moss, H., Leslie, D., Beck, D., Gonzalez, J., & Rayson, P. (2020). Boss: Bayesian optimization over string spaces. Advances in neural information processing systems, 33, 15476-15486.

import math
import itertools
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval

def pad(s, length):
    """
    Pad out input strings to a maximum length 
    (required to pass same length tensors to gpytorch modules)
    """
    new_s = torch.zeros(length, dtype=torch.double)
    new_s[:len(s)] = torch.tensor(s)
    return new_s

def build_one_hot(alphabet):
    """
    Build one-hot encodings for a given alphabet.
    """
    dim = len(alphabet)
    embs = torch.zeros((dim+1, dim), dtype=torch.double)
    index = {}
    for i, symbol in enumerate(alphabet):
        embs[i+1, i] = 1.0
        index[symbol] = i+1
    return embs, index

def encode_string(s, index):
    """
    Transform a string in a list of integers.
    The ints correspond to indices in an
    embeddings matrix.
    """
    return [index[symbol] for symbol in s]

class SubsequenceStringKernel(Kernel):
        def __init__(self, embds, index, alphabet=[], 
                    maxlen=80, batch_size=1000, _gap_decay=0.5, 
                    _match_decay=0.2, _order_coefs=[1/(2**i) for i in range(5)], 
                    normalize=True, **kwargs):
            super().__init__(**kwargs)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tensor_kwargs = {"dtype": torch.float, "device": device}
            # setting up hyper-parameters of string kernel
            self.register_parameter(
                name="raw_gap_decay",
                parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1) * _gap_decay),
            )
            raw_gap_decay_constraint = Interval(0, 1)
            self.register_constraint("raw_gap_decay", raw_gap_decay_constraint)

            self.register_parameter(
                name="raw_match_decay",
                parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1) * _match_decay),
            )
            raw_match_decay_constraint = Interval(0, 1)
            self.register_constraint("raw_match_decay", raw_match_decay_constraint)

            self.register_parameter(
                name="raw_order_coefs",
                parameter=torch.nn.Parameter(torch.tensor(_order_coefs) * torch.ones(*self.batch_shape, len(_order_coefs))),
            )
            raw_order_coefs_constraint = Interval(0, 1)
            self.register_constraint("raw_order_coefs", raw_order_coefs_constraint)

            self.alphabet = alphabet
            self.normalize = normalize
            self.embds = embds
            self.index = index
            self.embs_dim = self.embds.shape[1]
            self.maxlen = maxlen

            self.batch_size = batch_size

        @property
        def gap_decay(self) -> torch.Tensor:
            return self.raw_gap_decay_constraint.transform(self.raw_gap_decay)

        @gap_decay.setter
        def gap_decay(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_gap_decay=self.raw_gap_decay_constraint.inverse_transform(value))

        @property
        def match_decay(self) -> torch.Tensor:
            return self.raw_match_decay_constraint.transform(self.raw_match_decay)

        @match_decay.setter
        def match_decay(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_match_decay=self.raw_match_decay_constraint.inverse_transform(value))

        @property
        def order_coefs(self) -> torch.Tensor:
            return self.raw_order_coefs_constraint.transform(self.raw_order_coefs)

        @order_coefs.setter
        def order_coefs(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_order_coefs=self.raw_order_coefs_constraint.inverse_transform(value))


        def forward(self, 
                    X1:torch.Tensor,
                    X2:torch.Tensor, 
                    diag: bool = False, 
                    **params):
            if params.get("last_dim_is_batch", False):
                raise NotImplementedError("last_dim_is_batch not supported")
            # expanding batch shapes manually 
            # (kernel computation doesn't support auto-broadcasting)
            if X1.dim() > X2.dim():
                X1_shape_diff = X1.shape[:X1.dim()-X2.dim()]
                X2 = X2.expand(X1_shape_diff + X2.shape)
            if X2.dim() > X2.dim():
                X2_shape_diff = X2.shape[:X2.dim()-X1.dim()]
                X1 = X1.expand(X2_shape_diff + X1.shape)
            
            kernel_shape = X1.shape[:-2] + (X1.shape[-2],) + (X2.shape[-2],)
            K = torch.zeros(kernel_shape, **self.tensor_kwargs)
            if K.dim() > 2:
                for batch_idx in range(K.shape[0]):
                    K[batch_idx, :, :] = self._compute_kernel(X1[batch_idx], X2[batch_idx])
            else:
                K = self._compute_kernel(X1, X2, **params)
            if diag is True:
                return torch.diag(K)
            return K 

        def _compute_kernel(self, 
                            X1:torch.Tensor, 
                            X2:torch.Tensor, 
                            **params):
            self.D = self._precalc()
            if self.normalize:
                X1_diag_Ks = self._diag_calculations(X1)
                X2_diag_Ks = self._diag_calculations(X2)
            k_mat = torch.zeros((len(X1), len(X2)), **self.tensor_kwargs)
            tuples = list(itertools.product(range(X1.shape[0]), range(X2.shape[0])))
            num_batches = math.ceil(len(tuples)/self.batch_size)
            for i in range(num_batches):
                tuples_batch = tuples[self.batch_size*i:self.batch_size*(i+1)]
                X1_batch_indicies = torch.tensor([t[0] for t in tuples_batch], device=self.tensor_kwargs["device"])
                X2_batch_indicies = torch.tensor([t[1] for t in tuples_batch],  device=self.tensor_kwargs["device"])
                X1_batch = X1.index_select(dim=0, index=X1_batch_indicies)
                X2_batch = X2.index_select(dim=0, index=X2_batch_indicies)
                k_result = self._k(X1_batch, X2_batch)
                for j in range(0, len(tuples_batch)):
                    if self.normalize and X1_diag_Ks[tuples_batch[j][0]] != 0 and X2_diag_Ks[tuples_batch[j][1]] != 0:
                        k_result_norm = self._normalize(k_result[j], X1_diag_Ks[tuples_batch[j][0]], X2_diag_Ks[tuples_batch[j][1]])
                        k_mat[tuples_batch[j][0], tuples_batch[j][1]] = k_result_norm
                    else:
                        k_mat[tuples_batch[j][0], tuples_batch[j][1]] = k_result[j]
            return k_mat

        def _k(self, s1:torch.Tensor, s2:torch.Tensor):
            """
            Computes subsequence string kernel between two tensors of strings
            represented in numerical embeddings (one-hot encoding) from the alphabet.
            """
            S = torch.bmm(self.embds[s1.long()], self.embds[s2.long()].transpose(-2, -1))
            assert S.shape == (s1.shape[0], s1.shape[1], s1.shape[1])
            assert S.shape == (s1.shape[0], self.maxlen, self.maxlen)
            Kp = []
            Kp.append(torch.ones((s1.shape[0], self.maxlen, self.maxlen), **self.tensor_kwargs))
            match_sq = self.match_decay * self.match_decay
            for i in range(len(self.order_coefs)-1):
                aux = S * Kp[i]
                aux2 = aux @ self.D
                aux = aux2 * match_sq
                aux = aux.transpose(-2, -1) @ self.D
                Kp.append(aux.transpose(-2, -1))

            Kp = torch.cat([x.unsqueeze(0) for x in Kp], dim=0)
            final_aux1 = S * Kp
            final_aux2 = torch.sum(final_aux1, dim=-1)
            final_aux3 = torch.sum(final_aux2, dim=-1, keepdims=True)
            Ki = match_sq * final_aux3
            Ki = Ki.squeeze(-1)
            k = self.order_coefs @ Ki
            return k

        def _normalize(self, 
                       K_result:torch.Tensor, 
                       diag_Ks_i:torch.Tensor, 
                       diag_Ks_j:torch.Tensor):
            """
            Normalize the kernel.
            """
            norm = diag_Ks_i * diag_Ks_j
            sqrt_norm = torch.sqrt(norm)

            K_norm = K_result / sqrt_norm
            return K_norm

        def _diag_calculations(self, X:torch.Tensor):
            """
            Calculate the K(x,x) values first because
            they are used in normalization.
            """
            k_result = torch.zeros(len(X), **self.tensor_kwargs)
            num_batches = math.ceil(len(X)/self.batch_size)
            for i in range(num_batches):
                X_batch = X[self.batch_size*i:self.batch_size*(i+1),:]
                result = self._k(X_batch, X_batch)
                k_result[self.batch_size*i:self.batch_size*(i+1)] = result
            return k_result

        def _precalc(self):
            """
            Construct D: a upper triangular matrix over gap-decay powers.
            """
            tril = torch.tril(torch.ones((self.maxlen,self.maxlen), **self.tensor_kwargs))
            power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
            power = (torch.tensor(power, **self.tensor_kwargs).reshape(self.maxlen,self.maxlen) + tril)
            tril = (tril.T - torch.eye(self.maxlen, **self.tensor_kwargs))
            gaps = torch.ones([self.maxlen, self.maxlen], **self.tensor_kwargs)*self.gap_decay
            D = (gaps * tril) ** power
            return D