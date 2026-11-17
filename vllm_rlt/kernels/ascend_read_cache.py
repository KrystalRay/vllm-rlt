"""Bounded, write-through contiguous KV reads for one stable eager cohort.

Paged KV remains authoritative. This optional read cache never changes logical
coverage or resource ownership; missed/invalidated cohorts use paged attention.
Updates use ordinary clone/cat, with the same eager attention arithmetic.
"""

import weakref

import torch


class AscendKVReadCache:
    def __init__(self, layers, planes, heads, dim, dtype, device, pool_bytes, *, limit_bytes=None):
        self.layers, self.planes = layers, planes
        self.heads, self.dim, self.dtype, self.device = heads, dim, dtype, device
        self.pool_bytes = pool_bytes
        self.limit_bytes = limit_bytes
        self.enabled = True
        self.entries = {}
        self.bytes = 0
        self.budget = 0
        self.hits = self.misses = self.admission_rejections = 0
        self._key = self._rejected = self._metadata = self._signature = None
        self._context = None

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self._key = self._rejected = self._metadata = self._signature = None
        self._context = None

    def _prepare(self, batch):
        stream = (
            torch.npu.current_stream(self.device).stream_id if self.device.type == "npu" else None
        )
        signature = stream, batch.owner._allocation_epoch
        if (
            self._metadata is not None
            and self._metadata() is batch
            and self._signature == signature
        ):
            return self._context
        self._metadata = self._signature = None
        self._context = None
        ranges = batch.sequence_ranges
        if not self.enabled or not ranges:
            self.clear()
            return None
        start, count, depth = ranges[0].start, ranges[0].count, ranges[0].depth
        if any((r.start, r.count, r.depth) != (start, count, depth) for r in ranges):
            self.clear()
            return None
        key = signature, tuple((rid, id(a)) for rid, a in batch.allocations)
        if key != self._key and key != self._rejected:
            self.clear()
            reserved = sum(r.allocation.max_tokens for r in ranges)
            needed = reserved * self.heads * self.dim * (torch.finfo(self.dtype).bits // 8)
            needed *= 2 * self.layers * self.planes
            if self.limit_bytes is None:
                # Admit the WHOLE cohort, not an LRU that thrashes over layers.
                # Leave at least two thirds of currently free HBM for transient
                # attention/storage and overlapping lifetimes. No eager reserve.
                free, _ = torch.npu.mem_get_info(self.device)
                available = (
                    free
                    + torch.npu.memory_reserved(self.device)
                    - torch.npu.memory_allocated(self.device)
                )
                self.budget = min(16 << 30, self.pool_bytes, available // 3)
            else:
                self.budget = self.limit_bytes
            if needed <= self.budget:
                self._key = key
            else:
                self._rejected = key
                self.admission_rejections += 1
        if key == self._key:
            self._context = (len(ranges), start, count, depth)
        self._metadata = weakref.ref(batch)
        self._signature = signature
        return self._context

    def record(self, layer, batch, k, v):
        context = self._prepare(batch)
        if context is None:
            return
        rows, start, count, depth = context
        current = self.entries.get((depth, layer))
        key = k.reshape(rows, count, self.heads, self.dim).transpose(1, 2)
        value = v.reshape(rows, count, self.heads, self.dim).transpose(1, 2)
        if start == 0:
            # Own independent bytes: public write callers may mutate inputs.
            pair = key.clone(), value.clone()
        elif current is not None and current[0].shape[2] == start:
            pair = torch.cat((current[0], key), dim=2), torch.cat((current[1], value), dim=2)
        else:
            # No speculative bootstrap from potentially unwritten history.
            # Stay cold for this cohort until it writes a fresh position zero.
            self.entries.pop((depth, layer), None)
            if current is not None:
                self.bytes -= sum(x.numel() * x.element_size() for x in current)
            return
        old = 0 if current is None else sum(x.numel() * x.element_size() for x in current)
        new = sum(x.numel() * x.element_size() for x in pair)
        if self.bytes - old + new > self.budget:
            self.clear()
            return
        # Publish only after BOTH copies enqueue; no mutable output arena.
        self.entries[depth, layer] = pair
        self.bytes += new - old

    def get(self, layer, batch):
        context = self._prepare(batch)
        if context is not None:
            _, start, count, depth = context
            pair = self.entries.get((depth, layer))
            if pair is not None and pair[0].shape[2] == start + count:
                self.hits += 1
                return pair
        self.misses += 1
        return None
