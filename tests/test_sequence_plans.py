"""Range plans and contiguous reads preserve ownership and paged KV truth."""

import pytest
import torch

from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.kernels.ascend_read_cache import AscendKVReadCache
from vllm_rlt.worker.metadata import MetadataBuilder, stage_range_metadata


def cache_for(device="cpu", *, read_cache=False, incremental=False):
    device = torch.device(device)
    attention = (
        AttentionBackend(
            AscendEagerAttention(),
            BackendCapabilities(sequence_ranges=True),
            dict(selected_backend="torch", backend="torch"),
        )
        if device.type == "cpu"
        else None
    )
    cache = KVCacheManager(
        2,
        2,
        64,
        64,
        16,
        2,
        device=device,
        dtype=torch.bfloat16,
        backend="ascend_eager" if device.type == "npu" else "torch",
        attention=attention,
        incremental_allocation=incremental,
    )
    cache.key_cache.fill_(float("nan"))
    cache.value_cache.fill_(float("nan"))
    for rid in ("a", "b"):
        assert cache.allocate(rid, 35, initial_tokens=1 if incremental else None)
    if read_cache:
        cache._read_cache = AscendKVReadCache(
            2, 2, 2, 64, cache.dtype, cache.device, 1 << 25, limit_bytes=1 << 25
        )
    elif cache._read_cache:
        cache._read_cache.enabled = False
    return cache, MetadataBuilder(cache)


def compact(builder, start, count, depth=0):
    layouts = builder.build_ranges(["a", "b"], [start] * 2, [count] * 2, [[depth] * 2])
    return stage_range_metadata(layouts, builder.cache._stage)[0]


def test_plan_reuses_tables_without_reusing_mutable_positions():
    cache, builder = cache_for()
    first = compact(builder, 0, 17)
    later = compact(builder, 17, 1)
    assert first.table_plan is later.table_plan
    assert first.table_plan.bind(cache.device) is later.table_plan.bind(cache.device)
    assert first.table_plan.host.shape == (2, 3)
    assert first.position_ids.tolist() == list(range(17)) * 2
    assert later.position_ids.tolist() == [17, 17]
    legacy = cache._prepare_batch(["a"] * 17 + ["b"] * 17, [0] * 34, list(range(17)) * 2)
    assert tuple(first.rows) == legacy.rows
    assert torch.equal(first.write_blocks, legacy.write_blocks)
    assert torch.equal(first.write_offsets, legacy.write_offsets)
    cache.free("a")
    assert cache.allocate("a", 35)
    with pytest.raises(RuntimeError, match="stale"):
        cache._require_live_batch(first)
    assert compact(builder, 0, 1).table_plan is not first.table_plan


@pytest.mark.parametrize(
    "starts,counts,ids",
    [
        ([0, 1], [2, 2], ["a", "b"]),
        ([0, 0], [2, 3], ["a", "b"]),
        ([0, 0], [2, 2], ["a", "a"]),
        ([0, 0], [0, 0], ["a", "b"]),
        ([34, 34], [2, 2], ["a", "b"]),
    ],
)
def test_invalid_ranges_rejected(starts, counts, ids):
    _, builder = cache_for()
    with pytest.raises(ValueError):
        builder.build_ranges(ids, starts, counts, [[0, 0]])


def test_incremental_growth_invalidates_table_plan():
    cache, builder = cache_for(incremental=True)
    before = compact(builder, 0, 1)
    with pytest.raises(ValueError, match="physical"):
        compact(builder, 16, 1)
    assert cache.ensure_capacity("a", 17)
    assert cache.ensure_capacity("b", 17)
    after = compact(builder, 16, 1)
    assert before.table_plan is not after.table_plan


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@pytest.mark.parametrize("read_cache", [False, True])
@torch.inference_mode()
def test_range_attention_matches_paged_queries_and_preserves_outputs(device, read_cache):
    if device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
    torch.manual_seed(83)
    cache, builder = cache_for(device, read_cache=read_cache)
    held = snapshot = None
    for start, count in ((0, 17), (17, 9), (26, 1)):
        batch = compact(builder, start, count)
        # Direct tests have no runner upload; bind the owned sequence table.
        from dataclasses import replace

        batch = replace(batch, block_tables=batch.table_plan.bind(cache.device))
        ids = ["a"] * count + ["b"] * count
        positions = list(range(start, start + count)) * 2
        legacy = cache._prepare_batch(ids, [0] * len(ids), positions)
        k = torch.randn(len(ids), 2, 64, device=cache.device, dtype=cache.dtype)
        v = torch.randn_like(k)
        saved_k = k.clone()
        q = torch.randn(len(ids), 4, 128, device=cache.device, dtype=cache.dtype)[:, :, ::2]
        for layer in range(2):
            cache._write_prepared(layer, batch, k, v)
            actual = cache._attend_prepared(layer, batch, q)
            expected = cache.attention.kernel.attend_metadata(
                q, cache.key_cache[:, layer], cache.value_cache[:, layer], legacy
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert torch.isfinite(actual).all()
            if held is None:
                held, snapshot = actual, actual.clone()
        k.zero_()
        if read_cache:
            pair = cache._read_cache.get(0, batch)
            assert pair is not None
            torch.testing.assert_close(
                pair[0][:, :, -count:].transpose(1, 2).reshape_as(k), saved_k, atol=0, rtol=0
            )
    torch.testing.assert_close(held, snapshot, atol=0, rtol=0)
    if read_cache:
        assert cache._read_cache.hits >= 6
        cache.truncate_suffix("a", 0)
        assert not cache._read_cache.entries
        with pytest.raises(RuntimeError, match="uninitialized"):
            cache._attend_prepared(0, batch, q)


@torch.inference_mode()
def test_read_cache_admission_cold_start_and_holes():
    cache, builder = cache_for(read_cache=True)
    cache._read_cache.limit_bytes = 1
    batch = compact(builder, 0, 1)
    k = torch.randn(2, 2, 64, dtype=cache.dtype)
    cache._write_prepared(0, batch, k, k)
    assert not cache._read_cache.entries
    assert cache._read_cache.admission_rejections == 1
    cache._read_cache.limit_bytes = 1 << 25
    cache._read_cache.clear()
    # A nonzero first write cannot populate a read cache with unwritten history.
    cold = compact(builder, 2, 1)
    cache._write_prepared(0, cold, k, k)
    assert cache._read_cache.get(0, cold) is None
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache._attend_prepared(0, cold, k)
    first = compact(builder, 0, 1)
    cache._write_prepared(0, first, k, k)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache._attend_prepared(1, first, k)
    # Compatibility writes must invalidate the auxiliary copy.
    cache.write(0, ["a"], [0], [0], k[:1], k[:1])
    assert not cache._read_cache.entries


@pytest.mark.npu
@torch.inference_mode()
def test_sequence_tables_and_read_cache_do_not_cross_stream_ownership():
    from dataclasses import replace

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    cache, builder = cache_for("npu:0", read_cache=True)
    batch = compact(builder, 0, 17)
    k = torch.randn(34, 2, 64, device=cache.device, dtype=cache.dtype)
    q = torch.randn(34, 4, 64, device=cache.device, dtype=cache.dtype)
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(producer):
        first = replace(batch, block_tables=batch.table_plan.bind(cache.device))
        cache._write_prepared(0, first, k, k)
        held = cache._attend_prepared(0, first, q)
        snapshot = held.clone()
        ready = torch.npu.Event()
        ready.record()
    consumer.wait_event(ready)
    with torch.npu.stream(consumer):
        second = replace(batch, block_tables=batch.table_plan.bind(cache.device))
        assert first.block_tables.data_ptr() != second.block_tables.data_ptr()
        assert cache._read_cache.get(0, second) is None
        actual = cache._attend_prepared(0, second, q)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, snapshot, atol=0, rtol=0)
    torch.testing.assert_close(held, snapshot, atol=0, rtol=0)
    cache.free("a")
    cache.free("b")


def test_npu_cache_identity_never_resolves_raw_stream_handle(monkeypatch):
    """Identity lookups must not drain Torch-NPU's asynchronous task queue."""
    from types import SimpleNamespace

    from vllm_rlt.worker.metadata import RangeTablePlan

    class Stream:
        stream_id = 73

        @property
        def npu_stream(self):
            raise AssertionError("raw handle access drains the NPU submission queue")

    # No physical NPU required: all three cache-hit paths only need identity.
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda device: Stream()), raising=False
    )

    class Device:
        type = "npu"

    device = Device()
    assert AscendEagerAttention._stream_key(device) == 73
    plan = RangeTablePlan([[1]])
    sentinel = object()
    plan.tensors[device, 73] = sentinel
    assert plan.bind(device) is sentinel

    class Batch:
        owner = SimpleNamespace(_allocation_epoch=5)
        sequence_ranges = ()

    cache = AscendKVReadCache(1, 1, 1, 1, torch.bfloat16, device, 1024, limit_bytes=1024)
    assert cache._prepare(Batch()) is None
