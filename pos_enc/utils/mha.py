from typing import Callable, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from pos_enc.timing_utils import time_block


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class MultiheadAttention(torch.nn.Module):
    """
    Same as torch.nn.MultiheadAttention for bidirectional attention, but supports:
    - RMSNorm
    - customized attention function
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        predict_d : str = 'none', # none, predict_d, predict_dsig
        init_depth: float = 0.0,
        init_sigma: float = 3.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.predict_d = predict_d
        self.dropout = dropout
        self.bias = bias
        self.qk_norm = qk_norm

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if predict_d == 'predict_d':
            self.d_proj_weight = Parameter(torch.zeros((1, embed_dim)))
            self.d_proj_bias = Parameter(torch.tensor([0.0]))
        elif predict_d == 'predict_dsig':
            self.d_proj_weight = Parameter(torch.zeros((2, embed_dim)))
            self.d_proj_bias = torch.nn.Parameter(torch.tensor([init_depth, init_sigma]))
        else:
            self.register_parameter("d_proj_weight", None)
            self.register_parameter("d_proj_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        
        # if self.predict_d != 'none':
        #     constant_(self.d_proj_weight, 0.0)
        # if self.predict_d == 'predict_d':
        #     constant_(self.d_proj_bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sdpa_fn: Callable = F.scaled_dot_product_attention,
    ) -> Tensor:
        """
        Args:
            query: (B, T, D)
            key: (B, T, D)
            value: (B, T, D)
            sdpa_fn: (q, k, v, **kwargs) -> Tensor
        Returns:
            output: (B, T, D)
        """
        with time_block("mha_total", enabled=False):
            with time_block("get_features", enabled=False):
                if self.predict_d != 'none':
                    raw_d = F.linear(query, self.d_proj_weight, self.d_proj_bias)
                    # predicted_d = torch.exp(log_d) # (B, T, 1) or (B, T, 2)
                    # print(f"predicted logd max/min: {log_d.max()}/{log_d.min()}, predicted d max/min: {predicted_d.max()}/{predicted_d.min()}")
                    
                q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(
                    3, dim=0
                )
                if self.in_proj_bias is not None:
                    q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3, dim=0)
                else:
                    q_proj_bias = k_proj_bias = v_proj_bias = None

                q = F.linear(query, q_proj_weight, q_proj_bias)
                k = F.linear(key, k_proj_weight, k_proj_bias)
                v = F.linear(value, v_proj_weight, v_proj_bias)
                q, k, v = (
                    rearrange(x, "b t (h c) -> b t h c", h=self.num_heads) for x in [q, k, v]
                )
                if self.qk_norm:
                    q = self.q_norm(q)
                    k = self.k_norm(k)
                q, k, v = (rearrange(x, "b t h c -> b h t c") for x in [q, k, v])

            if self.predict_d != 'none':
                o = sdpa_fn(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                            predicted_d=raw_d)
            else:
                o = sdpa_fn(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            
            with time_block("get_features", enabled=True):
                o = rearrange(o, "b h t c -> b t (h c)", h=self.num_heads)
                attn_output = F.linear(o, self.out_proj.weight, self.out_proj.bias)

            return attn_output


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda:0"
    mha = MultiheadAttention(embed_dim=32, num_heads=8).to(device)

    q = torch.randn(10, 16, 32).to(device)
    k = torch.randn(20, 16, 32).to(device)
    v = torch.randn(20, 16, 32).to(device)

    output = mha(q, k, v, sdpa_fn=lambda q, k, v, **kwargs: v)
    print(output.size(), output.sum())
