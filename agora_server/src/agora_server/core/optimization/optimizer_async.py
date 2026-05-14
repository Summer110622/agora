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

import time

from agora_server.core.optimization.optimizer_sync import Optimizer

from hivemind.utils import get_logger


logger = get_logger(__name__)


class OptimizerAsync(Optimizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _resync_state_startup(self):
        if self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return True  # local gradients were computed with out-of-sync parameters, must start over
        else:
            with self.tracker.pause_updates():
                logger.log(
                    self.status_loglevel,
                    f"Catching up with collaboration step {self.tracker.global_epoch} at join phase",
                )
                self.state_averager.local_epoch = self.tracker.global_epoch
                self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)
        return False

    def _catchup_epoch(self) -> bool:
        # self.local_epoch+1, only catchup if behind by 2 steps - because behind by 1 will trigger averaging
        return (
            (self.tracker.global_epoch - self.max_allowed_stale) <= (self.local_epoch + 1) < self.tracker.global_epoch
        )

    def _auto_step(self) -> None:
        """Internal method to call `step()` automatically."""
        try:
            # Call the parent class's step method with one batch size
            logger.debug(f"AutoStepOptimizer step at {time.strftime('%H:%M:%S')}")
            self._last_step_time = time.time()

            # if delayed updates finished before step, apply these updates; otherwise do nothing
            self.state_averager.step(apply_delayed_updates=True)

            if self._resync_state():
                return None

            self._maybe_schedule_gradient_averaging()

            if self.in_update and self.tracker.ready_to_update_epoch:
                batch_size = 1
                self._step(batch_size=batch_size)

            self.state_averager.allow_state_sharing = self.allow_state_sharing
        except Exception as e:
            logger.error(f"Error in auto step: {e}")

    def load_state_from_peers(self, wait_for_end_round=False, **kwargs):
        # Wait for a while grad accumulation round before requesting state from peers
        logger.info("Waiting for peers in stage to finish step before joining")
        while True:
            if (
                self.tracker.fetched_global_progress_this_epoch.is_set()
                and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.1
            ):
                break
            else:
                logger.info("Waiting for peers in stage to finish step before joining")
                time.sleep(self.tracker.max_refresh_period)

        self._load_state_from_peers(**kwargs)

        if wait_for_end_round:
            self.tracker.fetched_global_progress_this_epoch.clear()
            while True:
                if (
                    self.tracker.fetched_global_progress_this_epoch.is_set()
                    and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.2
                ):
                    break
                else:
                    logger.info("Downloaded state, waiting for start of new round")
                    time.sleep(0.5)

        self._resync_state_startup()
        logger.info("Joining run")
