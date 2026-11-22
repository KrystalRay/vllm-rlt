# SPDX-License-Identifier: Apache-2.0
"""Optional ATen host dispatch for the standard Ouro decoder layer."""

from functools import cache, partial
from pathlib import Path

from torch import nn

from vllm_rlt.layers import RMSNorm
from vllm_rlt.models.ouro import OuroAttention, OuroDecoderLayer, OuroMLP


@cache
def load_ouro_layer_dispatch():
    from torch.utils.cpp_extension import load

    extension = load(
        name="vllm_rlt_ouro_layer",
        sources=[str(Path(__file__).parent / "csrc" / "ouro_layer.cpp")],
        extra_cflags=["-O3"],
        with_cuda=False,
    )
    return partial(_dispatch_layer, extension.forward)


def _standard_forward(module, expected_type):
    return (
        type(module) is expected_type
        and "forward" not in vars(module)
        and module._compiled_call_impl is None
        and not module._forward_pre_hooks
        and not module._forward_hooks
    )


def _dispatch_layer(native_forward, layer, hidden, rope, batch, kv):
    if not _standard_forward(layer, OuroDecoderLayer):
        return layer(hidden, rope, batch, kv)
    attn, mlp = layer.self_attn, layer.mlp
    if not (_standard_forward(attn, OuroAttention) and _standard_forward(mlp, OuroMLP)):
        return layer(hidden, rope, batch, kv)

    norms = (
        layer.input_layernorm,
        layer.input_layernorm_2,
        layer.post_attention_layernorm,
        layer.post_attention_layernorm_2,
    )
    projections = (
        attn.q_proj,
        attn.k_proj,
        attn.v_proj,
        attn.o_proj,
        mlp.gate_proj,
        mlp.up_proj,
        mlp.down_proj,
    )
    if not all(_standard_forward(norm, RMSNorm) for norm in norms) or not all(
        _standard_forward(proj, nn.Linear) and proj.bias is None for proj in projections
    ):
        return layer(hidden, rope, batch, kv)

    # Resolve parameters on each call so loading or replacing weights cannot
    # leave a stale native reference. Tensor subclasses keep Python dispatch.
    norm_weights = tuple(norm.weight for norm in norms)
    projection_weights = tuple(proj.weight for proj in projections)
    if not all(type(weight) is nn.Parameter for weight in norm_weights + projection_weights):
        return layer(hidden, rope, batch, kv)
    return native_forward(
        hidden,
        rope,
        batch,
        kv,
        norm_weights,
        projection_weights,
        tuple(norm.variance_epsilon for norm in norms),
        attn.config.head_dim,
        attn.layer_idx,
    )
