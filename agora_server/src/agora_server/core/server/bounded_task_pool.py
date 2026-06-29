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

"""A bounded, self-pruning drop-in replacement for hivemind's TaskPool.

This pool adds two worker-side guards (no trainer-side changes required):

  * Hard size bound (``pool_size``): ``submit_task`` does a non-blocking put and, when the
    queue is full, *rejects* the request (fails its future) instead of blocking. The
    rejection surfaces to the trainer as an RPC error, and the trainer's existing
    ban/retry routes the request elsewhere -- i.e. cooperative load-shedding. We must not
    block here: ``submit_task`` runs on the connection handler's asyncio loop, and a
    blocking ``put`` would freeze that handler for all its connections.

  * Stale-request drop (``stale_timeout``): a request older than ``stale_timeout`` is
    dropped when it reaches the front of the queue, since by then the trainer has almost
    certainly abandoned it (see ``stale_timeout`` note below). This avoids spending GPU
    compute on results nobody is waiting for; the hard bound above is what caps RAM.
"""

import time

from collections import namedtuple
from concurrent.futures import Future
from queue import Empty, Full

import torch

from hivemind.moe.server.task_pool import TaskPool
from hivemind.utils import get_logger
from hivemind.utils.mpfuture import InvalidStateError, MPFuture


logger = get_logger(__name__)

# Same shape as hivemind's Task (future, args) plus the submission time, so we can age out
# abandoned requests. Downstream code only touches ``.future``/``.args``, so this stays
# compatible with the base pool's input/output loops.
BoundedTask = namedtuple("BoundedTask", ("future", "args", "submit_time"))


class QueueFullError(RuntimeError):
    """Returned to the trainer when the worker's task queue is at capacity (load-shedding)."""


class BoundedTaskPool(TaskPool):
    """``TaskPool`` with a hard size bound (reject-on-full) and stale-request dropping."""

    def __init__(self, *args, stale_timeout: float | None = None, **kwargs):
        """
        Args:
            stale_timeout: drop queued requests older than this many seconds.
                Should be >= the trainer's request timeout for this direction. ``None`` disables stale-dropping.
            **kwargs: forwarded to ``TaskPool`` (incl. ``pool_size`` -> the queue bound).
        """
        super().__init__(*args, **kwargs)
        self.stale_timeout = stale_timeout
        # mp.Queue stores the configured cap here; used only for log/metric messages.
        self.max_queue_size = getattr(self.tasks, "_maxsize", 0)

    def submit_task(self, *args: torch.Tensor) -> Future:
        """Add a task to the bounded queue, returning a Future; reject (don't block) when full."""
        task = BoundedTask(MPFuture(), args, time.time())
        if self.get_task_size(task) > self.max_batch_size:
            task.future.set_exception(
                ValueError(f"Task size greater than max_batch_size ({self.max_batch_size}), it can't be processed")
            )
            return task.future
        try:
            self.tasks.put_nowait(task)
            self.undispatched_task_timestamps.put(task.submit_time)
        except Full:
            # Load-shed: fail fast so the trainer's ban/retry routes this request elsewhere.
            logger.warning(f"{self.name}: task queue full (cap={self.max_queue_size}), rejecting request")
            task.future.set_exception(QueueFullError(f"{self.name}: task queue full (cap={self.max_queue_size})"))
        return task.future

    def iterate_minibatches(self, *args, **kwargs):
        """Form minibatches like ``TaskPool``, but drop stale (abandoned) tasks on dequeue."""
        batch = []
        total_size = 0

        while True:
            if total_size >= self.min_batch_size and self.tasks.empty():
                yield batch
                batch = []
                total_size = 0
            try:
                task = self.tasks.get(timeout=self.timeout)
            except Empty:
                logger.warning(f"Timeout reached but batch doesn't contain >={self.min_batch_size} elements yet")
                continue

            if self.stale_timeout is not None and (time.time() - task.submit_time) > self.stale_timeout:
                # Consume the parallel timestamp so _pool_input_loop's pop-count stays balanced
                # (it pops one timestamp per *dispatched* task; we pop here for *dropped* ones).
                self.undispatched_task_timestamps.get()
                try:
                    task.future.set_exception(
                        TimeoutError(f"{self.name}: dropping stale request (age > {self.stale_timeout:.0f}s)")
                    )
                except InvalidStateError:
                    pass
                logger.warning(f"{self.name}: dropped stale task (age > {self.stale_timeout:.0f}s)")
                continue

            task_size = self.get_task_size(task)
            if total_size + task_size > self.max_batch_size:
                yield batch
                batch = []
                total_size = 0
            try:
                if task.future.set_running_or_notify_cancel():
                    batch.append(task)
                    total_size += task_size
            except InvalidStateError as e:
                logger.debug(f"Failed to add task to batch: {task.future} raised {e}")
