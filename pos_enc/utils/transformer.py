"""Modified from torch transformer impl. that supports:

- sdpa_fn: customized function for scaled-dot-product-attention calculation
- checkpointing: whether or now enable gradient checkpointing
- adaLN-Zero conditioning
"""

# mypy: allow-untyped-defs
import copy
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Sequential

from .config import InstantiateConfig
from .rayrope_mha import MultiheadAttention

__all__ = [
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "TransformerEncoderLayerConfig",
    "TransformerEncoderLayer",
    "TransformerDecoderLayerConfig",
    "TransformerDecoderLayer",
]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


sdpa_fn_default = F.scaled_dot_product_attention


@dataclass
class TransformerLayerConfig:
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: Callable[[Tensor], Tensor] = F.relu
    layer_norm_eps: float = 1e-5
    batch_first: bool = False
    norm_first: bool = False
    bias: bool = True
    elementwise_affine: bool = True
    qk_norm: bool = False
    predict_d: str = 'none'  # none, predict_d, predict_dsig


@dataclass
class TransformerEncoderLayerConfig(TransformerLayerConfig, InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerEncoderLayer)

    norm_type: Literal["layer_norm", "adaLN-Zero"] = "layer_norm"
    modulation_activation: Optional[Callable[[Tensor], Tensor]] = None


@dataclass
class TransformerDecoderLayerConfig(TransformerLayerConfig, InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerDecoderLayer)


@dataclass
class TransformerEncoderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerEncoder)
    layer: TransformerEncoderLayerConfig = field(
        default_factory=TransformerEncoderLayerConfig
    )
    num_layers: int = 24
    input_norm: bool = False
    output_norm: bool = False
    checkpointing: bool = False


@dataclass
class TransformerDecoderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerDecoder)
    layer: TransformerDecoderLayerConfig = field(
        default_factory=TransformerDecoderLayerConfig
    )
    num_layers: int = 24
    input_norm: bool = False
    output_norm: bool = False
    checkpointing: bool = False


class TransformerEncoder(Module):

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layer = cfg.layer.setup()
        self.layers = _get_clones(encoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        self.in_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.input_norm
            else None
        )
        self.out_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.output_norm
            else None
        )
        if cfg.layer.norm_type == "adaLN-Zero":
            assert (
                cfg.layer.modulation_activation is not None
            ), "modulation_activation must be provided for adaLN-Zero"
            assert (
                cfg.layer.norm_first
            ), "only norm_first=True is supported for adaLN-Zero"
            self.final_modulation_mlp = Sequential(
                cfg.layer.modulation_activation,
                Linear(cfg.layer.d_model, 2 * cfg.layer.d_model, bias=cfg.layer.bias),
            )
            self.final_norm = LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )

        self.checkpointing = cfg.checkpointing

    def forward(
        self,
        src: Tensor,
        sdpa_fn: Callable = sdpa_fn_default,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        output = src

        if self.in_norm is not None:
            output = self.in_norm(output)

        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    mod,
                    output,
                    sdpa_fn=sdpa_fn,
                    cond=cond,
                    use_reentrant=False,
                )
            else:
                output = mod(output, sdpa_fn=sdpa_fn, cond=cond)

        if self.out_norm is not None:
            output = self.out_norm(output)

        if self.cfg.layer.norm_type == "adaLN-Zero":
            shift, scale = self.final_modulation_mlp(cond).chunk(2, dim=-1)
            output = modulate(self.final_norm(output), shift, scale)

        return output


class TransformerDecoder(Module):

    def __init__(self, cfg: TransformerDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        decoder_layer = cfg.layer.setup()
        self.layers = _get_clones(decoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        self.in_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.input_norm
            else None
        )
        self.out_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.output_norm
            else None
        )
        self.checkpointing = cfg.checkpointing

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        sdpa_fn: Callable = sdpa_fn_default,
    ) -> Tensor:
        output = tgt

        if self.in_norm is not None:
            output = self.in_norm(output)

        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    mod, output, memory, sdpa_fn=sdpa_fn, use_reentrant=False
                )
            else:
                output = mod(output, memory, sdpa_fn=sdpa_fn)

        if self.out_norm is not None:
            output = self.out_norm(output)

        return output


class TransformerEncoderLayer(Module):

    def __init__(self, cfg: TransformerEncoderLayerConfig) -> None:
        self.cfg = cfg
        super().__init__()
        if cfg.norm_type == "adaLN-Zero":
            assert (
                cfg.modulation_activation is not None
            ), "modulation_activation must be provided for adaLN-Zero"
            assert cfg.norm_first, "only norm_first=True is supported for adaLN-Zero"

        self.self_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
            predict_d=cfg.predict_d,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias)
        self.dropout = Dropout(cfg.dropout)
        self.linear2 = Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias)

        self.norm_first = cfg.norm_first
        self.norm1 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm2 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.dropout1 = Dropout(cfg.dropout)
        self.dropout2 = Dropout(cfg.dropout)

        self.activation = cfg.activation
        self.norm_type = cfg.norm_type
        if cfg.norm_type == "adaLN-Zero":
            self.modulation_mlp = Sequential(
                cfg.modulation_activation,
                Linear(cfg.d_model, 6 * cfg.d_model, bias=cfg.bias),
            )

    def forward(
        self,
        src: Tensor,
        sdpa_fn: Callable = sdpa_fn_default,
        cond: Optional[Tensor] = None,
    ) -> Tensor:

        assert (cond is None and self.norm_type == "layer_norm") or (
            cond is not None and self.norm_type == "adaLN-Zero"
        ), "cond must be None for layer_norm, and not None for adaLN-Zero"

        if self.norm_type == "adaLN-Zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation_mlp(cond).chunk(6, dim=1)
            )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            if self.norm_type == "adaLN-Zero":
                y = modulate(self.norm1(x), shift_msa, scale_msa)
                x = x + gate_msa.unsqueeze(1) * self._sa_block(y, sdpa_fn=sdpa_fn)
                x = x + gate_mlp.unsqueeze(1) * self._ff_block(
                    modulate(self.norm2(x), shift_mlp, scale_mlp)
                )
            else:
                x = x + self._sa_block(self.norm1(x), sdpa_fn=sdpa_fn)
                x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, sdpa_fn=sdpa_fn))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, sdpa_fn: Callable = sdpa_fn_default) -> Tensor:
        x = self.self_attn(x, x, x, sdpa_fn=sdpa_fn)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):

    def __init__(self, cfg: TransformerDecoderLayerConfig) -> None:
        self.cfg = cfg
        super().__init__()
        self.self_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
        )
        self.multihead_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias)
        self.dropout = Dropout(cfg.dropout)
        self.linear2 = Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias)

        self.norm_first = cfg.norm_first
        self.norm1 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm2 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm3 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.dropout1 = Dropout(cfg.dropout)
        self.dropout2 = Dropout(cfg.dropout)
        self.dropout3 = Dropout(cfg.dropout)

        self.activation = cfg.activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        sdpa_fn: Callable = sdpa_fn_default,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), sdpa_fn)
            x = x + self._mha_block(self.norm2(x), memory, sdpa_fn)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, sdpa_fn))
            x = self.norm2(x + self._mha_block(x, memory, sdpa_fn))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, sdpa_fn: Callable = sdpa_fn_default) -> Tensor:
        x = self.self_attn(x, x, x, sdpa_fn=sdpa_fn)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self, x: Tensor, mem: Tensor, sdpa_fn: Callable = sdpa_fn_default
    ) -> Tensor:
        x = self.multihead_attn(x, mem, mem, sdpa_fn=sdpa_fn)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda:0"

    encoder = TransformerEncoder(
        TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=128,
                nhead=8,
                dropout=0.1,
                bias=True,
                qk_norm=True,
            ),
            num_layers=2,
        )
    ).to(device)

    src = torch.randn(10, 16, 128).to(device)
    output = encoder(src)
    print(output.size(), output.sum())

    decoder = TransformerDecoder(
        TransformerDecoderConfig(
            layer=TransformerDecoderLayerConfig(
                d_model=128,
                nhead=8,
                dropout=0.1,
                bias=True,
                qk_norm=True,
            ),
            num_layers=2,
        )
    ).to(device)

    tgt = torch.randn(10, 16, 128).to(device)
    memory = torch.randn(10, 16, 128).to(device)
    output = decoder(tgt, memory)
    print(output.size(), output.sum())
