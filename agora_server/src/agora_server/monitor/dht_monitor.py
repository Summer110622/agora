# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import threading
import time

from time import perf_counter

from hivemind.dht.node import DHTNode
from hivemind.dht.protocol import DHTProtocol
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)

_rpc_store_count = 0
_rpc_find_count = 0
_store_time_total = 0.0
_store_time_count = 0
_get_time_total = 0.0
_get_time_count = 0
_last_log_time = time.time()
_counter_lock = threading.Lock()

_read_counter: collections.Counter = collections.Counter()
_write_counter: collections.Counter = collections.Counter()
_key_monitor_started = False
_key_monitor_lock = threading.Lock()


def _try_log_stats(current_time: float) -> None:
    """Log and reset all stats if 60s have elapsed. Must be called while holding _counter_lock."""
    global \
        _rpc_store_count, \
        _rpc_find_count, \
        _store_time_total, \
        _store_time_count, \
        _get_time_total, \
        _get_time_count, \
        _last_log_time
    if current_time - _last_log_time < 60:
        return
    avg_store_ms = (_store_time_total / _store_time_count * 1000) if _store_time_count > 0 else 0.0
    avg_get_ms = (_get_time_total / _get_time_count * 1000) if _get_time_count > 0 else 0.0
    logger.info(
        f"[DHT RPC Stats] Last 60s - rpc_store: {_rpc_store_count}, rpc_find: {_rpc_find_count}, "
        f"avg_store_latency: {avg_store_ms:.1f}ms, avg_get_latency: {avg_get_ms:.1f}ms"
    )
    _rpc_store_count = 0
    _rpc_find_count = 0
    _store_time_total = 0.0
    _store_time_count = 0
    _get_time_total = 0.0
    _get_time_count = 0
    _last_log_time = current_time


def _start_key_monitor_if_needed() -> None:
    """Start the key stats reporting thread if not already running. Safe to call repeatedly."""
    global _key_monitor_started
    if _key_monitor_started:
        return
    with _key_monitor_lock:
        if _key_monitor_started:
            return
        t = threading.Thread(target=_key_stats_loop, daemon=True, name="dht-key-monitor")
        t.start()
        _key_monitor_started = True


def _key_stats_loop() -> None:
    while True:
        time.sleep(60)
        _report_and_reset_key_stats()


def _report_and_reset_key_stats() -> None:
    """Snapshot, clear, and log the top-10 most read and written DHT keys."""
    with _counter_lock:
        top_reads = _read_counter.most_common(10)
        top_writes = _write_counter.most_common(10)
        _read_counter.clear()
        _write_counter.clear()

    reads_lines = "\n".join(f"  {count:>6}  {key}" for key, count in top_reads) if top_reads else "  none"
    writes_lines = "\n".join(f"  {count:>6}  {key}" for key, count in top_writes) if top_writes else "  none"
    logger.info(f"[DHT Key Stats] Top reads (last 60s):\n{reads_lines}")
    logger.info(f"[DHT Key Stats] Top writes (last 60s):\n{writes_lines}")


def patch_dht_protocol_logging():
    """Monkey patch DHTProtocol and DHTNode to monitor RPC call counts, store/get latency, and per-key access frequency."""
    from hivemind.p2p import P2PContext
    from hivemind.proto import dht_pb2

    original_rpc_store = DHTProtocol.rpc_store
    original_rpc_find = DHTProtocol.rpc_find
    original_store_many = DHTNode.store_many
    original_get_many = DHTNode.get_many

    logger.info(
        "[DHT Monitor] Patching DHTProtocol and DHTNode to monitor RPC calls, operation latency, and key access frequency"
    )

    async def counted_rpc_store(self, request: dht_pb2.StoreRequest, context: P2PContext) -> dht_pb2.StoreResponse:
        global _rpc_store_count
        with _counter_lock:
            _rpc_store_count += 1
            _try_log_stats(time.time())
        return await original_rpc_store(self, request, context)

    async def counted_rpc_find(self, request: dht_pb2.FindRequest, context: P2PContext) -> dht_pb2.FindResponse:
        global _rpc_find_count
        with _counter_lock:
            _rpc_find_count += 1
            _try_log_stats(time.time())
        return await original_rpc_find(self, request, context)

    async def timed_store_many(self, *args, **kwargs):
        global _store_time_total, _store_time_count
        _start_key_monitor_if_needed()
        keys = args[0] if args else kwargs.get("keys", ())
        t0 = perf_counter()
        result = await original_store_many(self, *args, **kwargs)
        with _counter_lock:
            _store_time_total += perf_counter() - t0
            _store_time_count += 1
            for key in keys:
                _write_counter[str(key)] += 1
            _try_log_stats(time.time())
        return result

    async def timed_counted_get_many(self, *args, **kwargs):
        global _get_time_total, _get_time_count
        _start_key_monitor_if_needed()
        keys = args[0] if args else kwargs.get("keys", ())
        t0 = perf_counter()
        result = await original_get_many(self, *args, **kwargs)
        with _counter_lock:
            _get_time_total += perf_counter() - t0
            _get_time_count += 1
            for key in keys:
                _read_counter[str(key)] += 1
            _try_log_stats(time.time())
        return result

    counted_rpc_store.__name__ = "rpc_store"
    counted_rpc_store.__qualname__ = "DHTProtocol.rpc_store"
    counted_rpc_find.__name__ = "rpc_find"
    counted_rpc_find.__qualname__ = "DHTProtocol.rpc_find"
    timed_store_many.__name__ = "store_many"
    timed_store_many.__qualname__ = "DHTNode.store_many"
    timed_counted_get_many.__name__ = "get_many"
    timed_counted_get_many.__qualname__ = "DHTNode.get_many"

    DHTProtocol.rpc_store = counted_rpc_store
    DHTProtocol.rpc_find = counted_rpc_find
    DHTNode.store_many = timed_store_many
    DHTNode.get_many = timed_counted_get_many

    logger.info("[DHT Monitor] RPC monitoring active - will log stats and top-10 key access every 60 seconds")
