"""Optional host executor; every model operation remains ordinary eager ATen."""

import hashlib
import os
import sysconfig
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn.modules import module as module_hooks

from vllm_rlt.layers.normalization import RMSNorm


@lru_cache(maxsize=1)
def load_native_plan():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "csrc" / "ascend_eager_plan.cpp"
    identity = (
        source.read_bytes()
        + str(torch.__version__).encode()
        + str(torch._C._GLIBCXX_USE_CXX11_ABI).encode()
        + str(sysconfig.get_config_var("SOABI")).encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:12]
    root = os.environ.get(
        "VLLM_RLT_NATIVE_BUILD_DIR", str(Path.home() / ".cache/vllm-rlt/native-eager")
    )
    build = Path(root) / digest
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="rlt_ascend_eager_" + digest,
        sources=[str(source)],
        build_directory=str(build),
        extra_cflags=["-O3", "-g0"],
        with_cuda=False,
        verbose=True,
    )


class NativeOuroExecutor:
    """Eager layer calls with current weights and conservative Python fallback.

    The extension stores no model tensors. Each call borrows the module's live
    parameters, and returns owned output/KV tensors. KV is published through its
    traversal owner only after a complete layer call has enqueued successfully.
    """

    def __init__(self):
        self.extension = load_native_plan()
        self.native_layers = 0
        self.fallbacks = Counter()

    def _fallback(self, reason):
        self.fallbacks[reason] += 1
        return None

    def layer(self, layer, hidden, position_embeddings, batch, traversal):
        from vllm_rlt.models.ouro import OuroAttention, OuroDecoderLayer, OuroMLP

        if (
            type(layer) is not OuroDecoderLayer
            or type(layer.self_attn) is not OuroAttention
            or type(layer.mlp) is not OuroMLP
        ):
            return self._fallback("module_schema")
        if type(hidden) is not torch.Tensor or torch._C._get_tracing_state():
            return self._fallback("tensor_or_tracing")
        if any(
            getattr(module_hooks, name, {})
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        ):
            return self._fallback("global_hooks")
        # Layer hooks run through its ordinary nn.Module call. Child callbacks,
        # custom forwards and compiled children retain the Python implementation.
        for child in layer.modules():
            if child is layer:
                continue
            if (
                child._forward_hooks
                or child._forward_pre_hooks
                or child._backward_hooks
                or child._backward_pre_hooks
                or "forward" in child.__dict__
                or getattr(child, "_compiled_call_impl", None) is not None
            ):
                return self._fallback("child_hooks_or_forward")
        a, m = layer.self_attn, layer.mlp
        norms = (
            layer.input_layernorm,
            layer.input_layernorm_2,
            layer.post_attention_layernorm,
            layer.post_attention_layernorm_2,
        )
        linears = (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj)
        if any(type(x) is not RMSNorm for x in norms) or any(
            type(x) is not torch.nn.Linear for x in linears
        ):
            return self._fallback("leaf_schema")
        if any(x.bias is not None for x in linears):
            return self._fallback("linear_bias")
        if any(x.variance_epsilon != norms[0].variance_epsilon for x in norms):
            return self._fallback("norm_epsilon")
        weights = [x.weight for x in (*norms, *linears)]
        if any(type(x) is not torch.nn.Parameter for x in weights):
            return self._fallback("parameter_schema")
        prepare = getattr(traversal, "prepare_native", None)
        if prepare is None:
            return self._fallback("paged_storage")
        inputs = prepare(a.layer_idx, hidden)
        if inputs is None:
            return self._fallback("append_geometry")
        if a.config.head_dim != inputs.head_dim:
            return self._fallback("head_geometry")
        output, key, value = self.extension.layer_forward(
            hidden,
            *position_embeddings,
            weights,
            [] if inputs.history is None else inputs.history,
            inputs.mask,
            inputs.batch_size,
            inputs.query_tokens,
            inputs.token_start,
            inputs.head_dim,
            inputs.kv_heads,
            norms[0].variance_epsilon,
        )
        traversal.commit_native(a.layer_idx, inputs, (key, value))
        self.native_layers += 1
        return output
