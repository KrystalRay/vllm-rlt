// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <array>
#include <utility>

namespace py = pybind11;
using at::Tensor;

namespace {

Tensor rms_norm(const Tensor& hidden, const Tensor& weight, double epsilon) {
    auto value = hidden.to(at::kFloat);
    auto variance = value.square().mean(-1, true);
    // Match Tensor + Python float, which uses add.Tensor with a wrapped scalar.
    auto inverse_rms = at::rsqrt(at::add(variance, at::native::wrapped_scalar_tensor(epsilon)));
    return weight * (value * inverse_rms).to(hidden.scalar_type());
}

Tensor apply_rope(const Tensor& value, const Tensor& cosine, const Tensor& sine) {
    auto left = value * cosine;
    auto halves = value.chunk(2, -1);
    return left + at::cat({-halves[1], halves[0]}, -1) * sine;
}

Tensor forward_layer(
    const Tensor& hidden,
    const std::pair<Tensor, Tensor>& rope,
    const py::object& batch,
    const py::object& cache,
    const std::array<Tensor, 4>& norms,
    const std::array<Tensor, 7>& projections,
    const std::array<double, 4>& epsilon,
    int64_t head_dim,
    int64_t layer_index) {
    Tensor q, k, v;
    {
        py::gil_scoped_release release;
        auto normalized = rms_norm(hidden, norms[0], epsilon[0]);
        const std::array<int64_t, 3> shape{hidden.size(0), -1, head_dim};
        q = at::linear(normalized, projections[0]).view(shape);
        k = at::linear(normalized, projections[1]).view(shape);
        v = at::linear(normalized, projections[2]).view(shape);
        q = apply_rope(q, rope.first, rope.second);
        k = apply_rope(k, rope.first, rope.second);
    }

    // Use the same prepared KV operations and backend as OuroAttention.
    cache.attr("_write_prepared")(layer_index, batch, k, v);
    auto attention = cache.attr("_attend_prepared")(layer_index, batch, q).cast<Tensor>();

    py::gil_scoped_release release;
    auto projected = at::linear(attention.reshape({hidden.size(0), -1}), projections[3]);
    auto state = hidden + rms_norm(projected, norms[1], epsilon[1]);
    auto mlp_input = rms_norm(state, norms[2], epsilon[2]);
    auto gate = at::linear(mlp_input, projections[4]);
    auto up = at::linear(mlp_input, projections[5]);
    static const auto silu_and_mul = c10::Dispatcher::singleton()
        .findSchemaOrThrow("vllm_rlt::silu_and_mul", "")
        .typed<Tensor(const Tensor&, const Tensor&)>();
    auto down = at::linear(silu_and_mul.call(gate, up), projections[6]);
    return state + rms_norm(down, norms[3], epsilon[3]);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward_layer);
}
