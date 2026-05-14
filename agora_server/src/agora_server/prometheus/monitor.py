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


import logging
import multiprocessing as mp
import queue
import re
import threading

from datetime import datetime
from functools import partial
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Any, Callable

from agora_server.prometheus.roles import BaseRoleMetrics, DirectMetricsConfig, PushGatewayConfig, WorkerRoleMetrics
from agora_server.utils.flops import (
    get_num_flop_per_sample,
    get_num_params,
    get_peak_flops,
)
from torch.nn import Module

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class BasePromMonitor(threading.Thread):
    """Base Prometheus Monitor class to parse logs and record metrics."""

    def __init__(
        self,
        role_name: str,
        role_raw: str,
        exported_role: str | None,
        metrics_factory: Callable[..., BaseRoleMetrics],
        gateway_url: str | None = None,
        instance: str | None = None,
        port: int | None = None,
        save_unknown_errors: bool = False,
        num_save_error: int = 10,
        error_dir_path: str = "/home/logs/",
        start_logs: bool = False,
        metrics: BaseRoleMetrics | None = None,
        **extra_push_kwargs,
    ):
        """Initialize Base Prometheus Monitor.

        Args:
            role_name (str): Name of the role instance (e.g., "worker_1").
            role_raw (str): Raw role name (e.g., "worker", "trainer", "agora" - needed for Pushgateway).
            exported_role (str | None): Role name to expose to external systems (e.g., Pushgateway grouping). If None, defaults to role_name.
            metrics_factory (Callable[..., BaseRoleMetrics]): Factory function to create metrics object.
            gateway_url (str | None, optional): Optional Pushgateway URL. Defaults to None.
            instance (str | None, optional): Instance name used in Pushgateway. Defaults to None.
            port (int | None, optional): Port used for scraping. Defaults to None.
            save_unknown_errors (bool, optional): When True, save unknown errors in logs to local files. Defaults to False.
            num_save_error (int, optional): Max number of saved error files. Defaults to 10.
            error_dir_path (str, optional): Directory path to save error logs. Defaults to "/home/logs/".
            start_logs (bool, optional): If True, start logging immediately; if False, wait for start_logging() call. Defaults to False.
            metrics (BaseRoleMetrics | None, optional): Pre-built metrics object. When provided, skips Prometheus
                server/gateway setup entirely. Useful for offline replay with a custom metrics sink. Defaults to None.
            extra_push_kwargs: Additional keyword arguments for PushGatewayConfig.
        """
        super().__init__()

        # How many errors to save
        self.save_unknown_errors = save_unknown_errors
        self.num_save_error = num_save_error
        self.save_error_cntr = 0

        self.start_logs = start_logs

        # When a pre-built metrics object is provided, skip all Prometheus infrastructure
        if metrics is not None:
            self.metrics = metrics
            self._log_queue = None
            self.queue_handler = None
            self._stop_event = None
            return

        # Create a log handler
        self._log_queue = mp.Queue()
        self.queue_handler = QueueHandler(self._log_queue)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.queue_handler.setFormatter(formatter)
        self._stop_event = threading.Event()

        # Create directory for error logs if needed
        if self.save_unknown_errors:
            log_dir = Path(error_dir_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.dump_unknown_error_func = partial(dump_unknown_error, dir_path=log_dir)

        # Create metrics
        if gateway_url:
            if not instance:
                logger.error("Instance name must be provided when using Push Gateway.")
                raise ValueError("Instance name must be provided when using Push Gateway.")

            push_config = PushGatewayConfig(
                gateway_url=gateway_url,
                instance=instance,
                **extra_push_kwargs,
            )
            direct_config = None
        elif port is not None:
            push_config = None
            direct_config = DirectMetricsConfig(port=port)
        else:
            logger.error("Either gateway_url or port must be provided for Prometheus monitoring.")
            raise ValueError("Either gateway_url or port must be provided for Prometheus monitoring.")

        self.metrics = metrics_factory(
            role_name=role_name,
            role_raw=role_raw,
            exported_role=exported_role,
            push_config=push_config,
            direct_config=direct_config,
        )

    def stop(self):
        """Signal the monitor thread to stop and wait for it to finish."""
        self._stop_event.set()
        self.metrics.shutdown()
        self.join(timeout=5)

    def start_logging(self, *args, **kwargs):
        """Start logging."""
        self.start_logs = True


class WorkerRegex:
    # Errors
    missed_ar_step: str = "error_missed_ar_step"
    load_timeout: str = "error_loadtimeout"
    regex_load_timeout: str = r".*Failed to load state from peers.*TimeoutError.*"
    grad_version: str = "error_grad_version"
    regex_grad_version: str = (
        r"\bone of the variables needed for gradient computation has been modified by an inplace operation\b"
    )
    unknown_error: str = "error_unknown"
    regex_unknown_error: str = r".*ERROR.*"

    # AR Runner Errors
    ar_stream_closed_early: str = "error_ar_stream_closed_early"
    regex_ar_stream_closed_early: str = r"Read stream closed early from sender"
    ar_accumulate_buffer_full: str = "error_ar_accumulate_buffer_full"
    regex_ar_accumulate_buffer_full: str = r"Accumulate buffer full, no sentinel added"

    # Gradients
    pre_max_grad: str = "pre_max_grad"
    regex_pre_max_grad: str = r"Pre Max total_grad:\s*([\d.]+)"
    pre_avg_grad: str = "pre_avg_grad"
    regex_pre_avg_grad: str = r"Pre Avg total_grad:\s*([\d.]+)"
    error_psgd_grad: str = "error_psgd_grad"
    regex_error_psgd_grad: str = r"Error norm val:\s*([\d.]+)"
    error_pre_psgd_grad: str = "error_pre_psgd_grad"
    regex_error_pre_psgd_grad: str = r"Error pre-clip norm val:\s*([\d.]+)"
    pre_psgd_grad: str = "pre_psgd_grad"
    regex_pre_psgd_grad: str = r"Prepsgd norm val:\s*([\d.]+)"
    post_psgd_grad: str = "post_psgd_grad"
    regex_post_psgd_grad: str = r"Postpsgd norm val:\s*([\d.]+)"
    pre_step_grad: str = "pre_step_grad"
    regex_pre_step_grad: str = r"Pre step total_grad:\s*([\d.]+)"
    post_step_grad: str = "post_step_grad"
    regex_post_step_grad: str = r"Post step total_grad:\s*([\d.]+)"
    # GPU info
    gpu_tot: str = "gpu_total"
    regex_gpu_tot: str = r"GPU mem size is (\d+)Gb"
    gpu_alloc: str = "gpu_alloc"
    regex_gpu_alloc: str = r"Allocated GPU mem is (0\.\d+)"
    gpu_res: str = "gpu_reserve"
    regex_gpu_res: str = r"Reserved GPU mem is (0\.\d+)"
    gpu_util: str = "gpu_util"
    regex_gpu_util: str = r".*GPU utilization is (\d+\.?\d*).*"
    # RAM info
    ram_tot: str = "ram_total"
    regex_ram_tot: str = r"Total RAM mem is (\d+)Gb"
    ram_use: str = "ram_usage"
    regex_ram_use: str = r"Used RAM mem is (0\.\d+)"
    # CPU info
    cpu_load: str = "cpu_load"
    regex_cpu_load: str = r"CPU load is (\d+\.\d+)"
    # Node static network info (latency / bandwidth, set once at startup)
    node_latency_ms: str = "latency_rtt"
    regex_node_latency_ms: str = r"Latency:\s*([\d.]+)\s*ms"
    node_download_mbps: str = "download_speed"
    regex_node_download_mbps: str = r"Download Speed:\s*([\d.]+)\s*Mbps"
    node_upload_mbps: str = "upload_speed"
    regex_node_upload_mbps: str = r"Upload Speed:\s*([\d.]+)\s*Mbps"
    # Communication
    peer_vis: str = "peer_vis"
    regex_peer_vis: str = r"see\s+(\d+)\s+of\s+([a-zA-Z0-9]+)\."
    send_comm: str = "send_comm"
    regex_send_comm: str = r"Sent\s+(\d+\.\d+)"
    recv_comm: str = "recv_comm"
    regex_recv_comm: str = r"Rcv\s+(\d+\.\d+)"
    tot_comm: str = "tot_comm"
    open_conns: str = "open_conns"
    regex_open_conns: str = r"Number open connections (-?\d+)"
    rpc_store: str = "rpc_store"
    rpc_find: str = "rpc_find"
    dht_store_latency: str = "dht_store_latency"
    dht_get_latency: str = "dht_get_latency"
    regex_rpc: str = r"\[DHT RPC Stats\] Last 60s - rpc_store: (\d+), rpc_find: (\d+), avg_store_latency: ([\d.]+)ms, avg_get_latency: ([\d.]+)ms"
    # Processed data
    batch_step: str = "batch_step"
    regex_batch_step: str = r"accumulated (\d+) samples"
    forward_batches: str = "forward_batches"
    regex_forward_batches: str = r".*\.\d+\.\d+_forward:\s*(\d+)"
    backward_batches: str = "backward_batches"
    regex_backward_batches: str = r".*\.\d+\.\d+_backward:\s*(\d+)"
    # Trainer saturation
    batch_await_time: str = "batch_await_time"
    avg_batch_size: str = "avg_batch_size"
    # MFU and FLOP
    mfu: str = "mfu"
    regex_forward_pattern: str = (
        r"([\d-]+ [\d:,]+).*?_forward:.*?(\d+)\s+examples.*?avg batch size ([\d.]+).*?avg batch await time (\d+)"
    )
    regex_backward_pattern: str = (
        r"([\d-]+ [\d:,]+).*?_backward:.*?(\d+)\s+examples.*?avg batch size ([\d.]+).*?avg batch await time (\d+)"
    )
    processed_pattern: str = r".*Processed (\d+) batches.*"
    # All-reduce and matchmaking grouping
    formed_groupsize: str = "formed_groupsize"
    max_groupsize: str = "max_groupsize"
    mm_time: str = "mm_time"
    regex_mm: str = r"Formed group with (\d+) peers out of (\d+) in ([\d.]+) secs"
    grad_avg_timeout: str = "grad_avg_timeout"
    grad_avg_groupsize: str = "grad_avg_groupsize"
    regex_grad_avg: str = r"INFO - Averaging gradients failed with TimeoutError\(\)"
    regex_state_avg: str = r"INFO - Averaging parameters failed with TimeoutError\(\)"
    state_avg_timeout: str = "state_avg_timeout"
    regex_grad_zero_group: str = (
        r"INFO - Averaging gradients failed with AllreduceException\('Averaging step failed: could not find a group'\)"
    )
    regex_grad_num_group: str = r"INFO - Averaged gradients with (\d+) peers"
    average_pcnt: str = "average_pcnt"
    regex_average_pcnt: str = r"Averaging: received (\d+(?:\.\d+)?)%"
    reducer_pcnt: str = "reducer_pcnt"
    regex_reducer_pcnt: str = r"Reducer: received (\d+(?:\.\d+)?)%"
    ar_peer_failure: str = "ar_peer_failure"
    regex_ar_peer_failure: str = r"Failed to receive tensors from peer \S+: \d+ failures, reason=(\w+)"
    ar_sender_banned: str = "ar_sender_banned"
    regex_ar_sender_banned: str = r"Banned sender \S+: reason=(\w+)"
    ar_request_rejected: str = "ar_request_rejected"
    regex_ar_request_rejected: str = r"Rejected request from \S+: reason=(\w+)"
    ar_yield_stall: str = "ar_yield_stall"
    regex_ar_yield_stall: str = (
        r"AR yield blocked ([0-9.]+)s for sender \d+ \(inbound_qsize=(\d+)/\d+\)"
    )
    ar_in_qsize: str = "ar_in_qsize"
    ar_barrier_wait: str = "ar_barrier_wait"
    regex_ar_barrier_wait: str = r"AR barrier wait ([0-9.]+)s on part \d+ sender \d+"
    ar_in_qsize_max: str = "ar_in_qsize_max"
    ar_in_qsize_mean: str = "ar_in_qsize_mean"
    regex_ar_queue_stats: str = (
        r"AR queue stats sender \d+: inbound max=(\d+)/\d+ mean=([0-9.]+)"
    )
    ar_chunk_gap_max: str = "ar_chunk_gap_max"
    ar_chunk_gap_mean: str = "ar_chunk_gap_mean"
    regex_ar_chunk_gap_stats: str = (
        r"AR chunk gap stats sender \d+: max=([0-9.]+)s mean=([0-9.]+)s n=\d+"
    )
    ar_time: str = "ar_time"
    grad_accum_time: str = "grad_accum_time"
    step: str = "step"
    regex_start_ar: str = r"([\d-]+ [\d:,]+) - INFO - Beginning optimizer step #(\d+)"
    regex_end_ar: str = r"([\d-]+ [\d:,]+) - INFO - Transitioning to epoch (\d+)"
    state_ar_time = "state_ar_time"
    regex_start_state_ar = r"([\d-]+ [\d:,]+) - INFO - All-reduce round started at local epoch #(\d+)"
    regex_end_state_ar = r"([\d-]+ [\d:,]+) - INFO - All-reduce round finished at local epoch #(\d+)"


class WorkerPromMonitor(BasePromMonitor):
    """Prometheus Monitor class for Worker role to parse logs and record metrics."""

    def __init__(
        self,
        role_name: str,
        role_raw: str = "worker",
        stats_report_interval: int = 60,
        exported_role: str | None = None,
        gateway_url: str | None = None,
        instance: str | None = None,
        port: int | None = None,
        save_unknown_errors: bool = False,
        num_save_error: int = 10,
        error_dir_path: str = "/home/logs/",
        start_logs: bool = False,
        metrics: BaseRoleMetrics | None = None,
        **extra_push_kwargs,
    ):
        """Initialize Worker Prometheus Monitor.

        Args:
            role_name (str): Name of the role instance (e.g., "worker_1").
            role_raw (str): Raw role name (e.g., "worker", "trainer", "agora" - needed for Pushgateway).
            stats_report_interval (int): Interval in seconds for reporting processed batches (used for MFU calculation).
            exported_role (str | None): Role name to expose to external systems (e.g., Pushgateway grouping). If None, defaults to role_name.
            gateway_url (str | None, optional): Optional Pushgateway URL. Defaults to None.
            instance (str | None, optional): Instance name used in Pushgateway. Defaults to None.
            port (int | None, optional): Port used for scraping. Defaults to None.
            save_unknown_errors (bool, optional): When True, save unknown errors in logs to local files. Defaults to False.
            num_save_error (int, optional): Max number of saved error files. Defaults to 10.
            error_dir_path (str, optional): Directory path to save error logs. Defaults to "/home/logs/".
            start_logs (bool, optional): If True, start logging immediately; if False, wait for start_logging() call. Defaults to False.
            metrics (BaseRoleMetrics | None, optional): Pre-built metrics sink for offline use. Defaults to None.
            extra_push_kwargs: Additional keyword arguments for PushGatewayConfig.
        """
        super().__init__(
            role_name=role_name,
            role_raw=role_raw,
            exported_role=exported_role,
            metrics_factory=WorkerRoleMetrics,
            gateway_url=gateway_url,
            instance=instance,
            port=port,
            save_unknown_errors=save_unknown_errors,
            num_save_error=num_save_error,
            error_dir_path=error_dir_path,
            start_logs=start_logs,
            metrics=metrics,
            **extra_push_kwargs,
        )

        # Values for computing flops.
        self.stats_report_interval = stats_report_interval
        self.flop_per_sample = None
        self.gpu_peak_flops = None
        self._has_metrics_sink = metrics is not None
        self.mfu_timestamp = None
        self.mfu_vals = []
        self.num_flop = []
        self.scaling = float(10**15)

        self.worker_regex = WorkerRegex()

        # Stateful vars
        self.begin_time = None
        self.begin_epoch = None
        self.trans_time = None
        self.trans_epoch = None
        self.state_ar_begin_time = None

    def start_logging(
        self,
        model: Module,
        model_conf: Any,
        use_mixed_precision: bool = False,
        enable_tf32: bool = False,
    ):
        """Get required values from model config and start logging."""
        num_params = get_num_params(model, exclude_embedding=True)
        self.flop_per_sample = get_num_flop_per_sample(num_params, model_conf)

        if use_mixed_precision:
            precision = "bf16"
        elif enable_tf32:
            precision = "tf32"
        else:
            precision = "fp32"

        if not self._has_metrics_sink:
            self.gpu_peak_flops = get_peak_flops(precision=precision)

        self.start_logs = True

    def process_log_entry(self, log_entry):
        """Process a single log entry, applying regex patterns and recording metrics.

        Can be called directly for offline/replay processing.
        """
        # Errors
        match = re.search(self.worker_regex.regex_load_timeout, log_entry, re.DOTALL)
        if match:
            self.metrics.record_error(self.worker_regex.load_timeout)
            return

        match = re.search(self.worker_regex.regex_grad_version, log_entry)
        if match:
            self.metrics.record_error(self.worker_regex.grad_version)
            return

        match = re.search(self.worker_regex.regex_ar_accumulate_buffer_full, log_entry)
        if match:
            self.metrics.record_error(self.worker_regex.ar_accumulate_buffer_full)
            return

        match = re.search(self.worker_regex.regex_ar_stream_closed_early, log_entry)
        if match:
            self.metrics.record_error(self.worker_regex.ar_stream_closed_early)
            return

        match = re.search(self.worker_regex.regex_unknown_error, log_entry)
        if match:
            self.metrics.record_error(self.worker_regex.unknown_error)

            # If unknown, save to a file
            if self.save_error_cntr >= self.num_save_error or not self.save_unknown_errors:
                return

            if self.dump_unknown_error_func(log_entry):
                self.save_error_cntr += 1

            return

        # Gradients information
        match = re.search(self.worker_regex.regex_pre_max_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.pre_max_grad}", val)
            return

        match = re.search(self.worker_regex.regex_pre_avg_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.pre_avg_grad}", val)
            return

        match = re.search(self.worker_regex.regex_error_psgd_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.error_psgd_grad}", val)
            return

        match = re.search(self.worker_regex.regex_error_pre_psgd_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.error_pre_psgd_grad}", val)
            return

        match = re.search(self.worker_regex.regex_pre_psgd_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.pre_psgd_grad}", val)
            return

        match = re.search(self.worker_regex.regex_post_psgd_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.post_psgd_grad}", val)
            return

        match = re.search(self.worker_regex.regex_pre_step_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.pre_step_grad}", val)
            return

        match = re.search(self.worker_regex.regex_post_step_grad, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.post_step_grad}", val)
            return

        # GPU
        match = re.search(self.worker_regex.regex_gpu_tot, log_entry)
        if match:
            val = match.group(1)
            self.metrics.record_operation(f"{self.worker_regex.gpu_tot}", val)
            return

        match = re.search(self.worker_regex.regex_gpu_alloc, log_entry)
        if match:
            val = match.group(1)
            self.metrics.record_operation(f"{self.worker_regex.gpu_alloc}", val)
            return

        match = re.search(self.worker_regex.regex_gpu_res, log_entry)
        if match:
            val = match.group(1)
            self.metrics.record_operation(f"{self.worker_regex.gpu_res}", val)
            return

        match = re.search(self.worker_regex.regex_gpu_util, log_entry)
        if match:
            gpu_util = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.gpu_util}", gpu_util)
            return

        # RAM
        match = re.search(self.worker_regex.regex_ram_tot, log_entry)
        if match:
            val = match.group(1)
            self.metrics.record_operation(f"{self.worker_regex.ram_tot}", val)
            return

        match = re.search(self.worker_regex.regex_ram_use, log_entry)
        if match:
            val = match.group(1)
            self.metrics.record_operation(f"{self.worker_regex.ram_use}", val)
            return

        # CPU
        match = re.search(self.worker_regex.regex_cpu_load, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.cpu_load}", val)
            return

        # Node static network info
        match = re.search(self.worker_regex.regex_node_latency_ms, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.node_latency_ms}", val)
            return

        match = re.search(self.worker_regex.regex_node_download_mbps, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.node_download_mbps}", val)
            return

        match = re.search(self.worker_regex.regex_node_upload_mbps, log_entry)
        if match:
            val = float(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.node_upload_mbps}", val)
            return

        # Communication
        match = re.search(self.worker_regex.regex_peer_vis, log_entry)
        if match:
            val = match.group(1)
            uid = match.group(2)
            self.metrics.record_operation(f"{self.worker_regex.peer_vis}_{uid}", val)
            return

        match1 = re.search(self.worker_regex.regex_send_comm, log_entry)
        match2 = re.search(self.worker_regex.regex_recv_comm, log_entry)
        if match1 and match2:
            send_val = float(match1.group(1))
            recv_val = float(match2.group(1))
            self.metrics.record_operation(self.worker_regex.send_comm, send_val)
            self.metrics.record_operation(self.worker_regex.recv_comm, recv_val)
            self.metrics.record_operation(self.worker_regex.tot_comm, send_val + recv_val)
            return

        match = re.search(self.worker_regex.regex_open_conns, log_entry)
        if match:
            open_conns = int(match.group(1))
            self.metrics.record_operation(self.worker_regex.open_conns, open_conns)
            return

        match = re.search(self.worker_regex.regex_rpc, log_entry)
        if match:
            self.metrics.record_operation(self.worker_regex.rpc_store, int(match.group(1)))
            self.metrics.record_operation(self.worker_regex.rpc_find, int(match.group(2)))
            self.metrics.time_operation(self.worker_regex.dht_store_latency, float(match.group(3)))
            self.metrics.time_operation(self.worker_regex.dht_get_latency, float(match.group(4)))
            return

        # Processed data
        match = re.search(self.worker_regex.regex_batch_step, log_entry)
        if match:
            accumulated_samples = int(match.group(1))
            self.metrics.record_operation(f"{self.worker_regex.batch_step}", accumulated_samples)
            return

        match = re.search(self.worker_regex.regex_forward_batches, log_entry)
        if match:
            batch_num = int(match.group(1))
            self.metrics.record_operation(self.worker_regex.forward_batches, batch_num)
            # Do not return, to allow MFU matching in same log line

        match = re.search(self.worker_regex.regex_backward_batches, log_entry)
        if match:
            batch_num = int(match.group(1))
            self.metrics.record_operation(self.worker_regex.backward_batches, batch_num)
            # Do not return, to allow MFU matching in same log line

        # MFU and trainer saturation
        forward_match = re.search(self.worker_regex.regex_forward_pattern, log_entry)
        if forward_match:
            self.metrics.record_operation(self.worker_regex.avg_batch_size, float(forward_match.group(3)))
            self.metrics.time_operation(self.worker_regex.batch_await_time, float(forward_match.group(4)))
            if self.flop_per_sample:
                timestamp = forward_match.group(1)

                if self.mfu_timestamp is None:
                    self.mfu_timestamp = timestamp

                if calc_time_diff(self.mfu_timestamp, timestamp) > 5000:
                    self.metrics.record_operation(self.worker_regex.mfu, sum(self.mfu_vals))
                    self.mfu_vals = []
                    self.mfu_timestamp = timestamp

                forward_examples = int(forward_match.group(2))
                forward_num_flop = forward_examples * self.flop_per_sample.forward
                forward_flops = forward_num_flop / self.stats_report_interval
                self.num_flop.append(forward_num_flop / self.scaling)

                if self.gpu_peak_flops:
                    mfu = forward_flops / self.gpu_peak_flops
                    self.mfu_vals.append(mfu)

            return

        backward_match = re.search(self.worker_regex.regex_backward_pattern, log_entry)
        if backward_match:
            self.metrics.record_operation(self.worker_regex.avg_batch_size, float(backward_match.group(3)))
            self.metrics.time_operation(self.worker_regex.batch_await_time, float(backward_match.group(4)))
            if self.flop_per_sample:
                timestamp = backward_match.group(1)

                if self.mfu_timestamp is None:
                    self.mfu_timestamp = timestamp

                if calc_time_diff(self.mfu_timestamp, timestamp) > 5000:
                    self.metrics.record_operation(self.worker_regex.mfu, sum(self.mfu_vals))
                    self.mfu_vals = []
                    self.mfu_timestamp = timestamp

                backward_examples = int(backward_match.group(2))
                backward_num_flop = backward_examples * self.flop_per_sample.backward
                backward_flops = backward_num_flop / self.stats_report_interval
                self.num_flop.append(backward_num_flop / self.scaling)

                if self.gpu_peak_flops:
                    mfu = backward_flops / self.gpu_peak_flops
                    self.mfu_vals.append(mfu)

            return

        # FLOP
        processed_match = re.search(self.worker_regex.processed_pattern, log_entry)
        if processed_match and len(self.num_flop) > 0:
            self.metrics.record_flop(sum(self.num_flop))
            self.num_flop = []
            return

        # All-reduce and matchmaking grouping
        match = re.search(self.worker_regex.regex_mm, log_entry)
        if match:
            num_peers = int(match.group(1))
            max_peers = int(match.group(2))
            seconds = float(match.group(3))
            self.metrics.record_operation(self.worker_regex.formed_groupsize, num_peers)
            self.metrics.record_operation(self.worker_regex.max_groupsize, max_peers)
            self.metrics.record_operation(self.worker_regex.mm_time, seconds)
            return

        match = re.search(self.worker_regex.regex_grad_avg, log_entry)
        if match:
            self.metrics.record_operation(self.worker_regex.grad_avg_timeout, 1)
            self.metrics.record_operation(self.worker_regex.grad_avg_groupsize, 0)
            return

        match = re.search(self.worker_regex.regex_state_avg, log_entry)
        if match:
            self.metrics.record_operation(self.worker_regex.state_avg_timeout, 1)
            return

        match = re.search(self.worker_regex.regex_grad_zero_group, log_entry)
        if match:
            self.metrics.record_operation(self.worker_regex.grad_avg_groupsize, 0)
            return

        match = re.search(self.worker_regex.regex_grad_num_group, log_entry)
        if match:
            num_peers = int(match.group(1))
            self.metrics.record_operation(self.worker_regex.grad_avg_groupsize, num_peers)
            self.metrics.record_operation(self.worker_regex.grad_avg_timeout, 0)
            return

        match = re.search(self.worker_regex.regex_average_pcnt, log_entry)
        if match:
            percentage = float(match.group(1))
            self.metrics.record_operation(self.worker_regex.average_pcnt, percentage)
            return

        match = re.search(self.worker_regex.regex_reducer_pcnt, log_entry)
        if match:
            percentage = float(match.group(1))
            self.metrics.record_operation(self.worker_regex.reducer_pcnt, percentage)
            return

        match = re.search(self.worker_regex.regex_ar_peer_failure, log_entry)
        if match:
            reason = match.group(1)
            self.metrics.record_error(f"{self.worker_regex.ar_peer_failure}_{reason}")
            return

        match = re.search(self.worker_regex.regex_ar_sender_banned, log_entry)
        if match:
            reason = match.group(1)
            self.metrics.record_error(f"{self.worker_regex.ar_sender_banned}_{reason}")
            return

        match = re.search(self.worker_regex.regex_ar_request_rejected, log_entry)
        if match:
            reason = match.group(1)
            self.metrics.record_error(f"{self.worker_regex.ar_request_rejected}_{reason}")
            return

        # State averaging all-reduce timing (async SPARTA AR communication)
        match = re.search(self.worker_regex.regex_start_state_ar, log_entry)
        if match:
            self.state_ar_begin_time = match.group(1)
            return

        match = re.search(self.worker_regex.regex_ar_yield_stall, log_entry)
        if match:
            duration = float(match.group(1))
            inbound_q = int(match.group(2))
            self.metrics.time_operation(self.worker_regex.ar_yield_stall, duration)
            self.metrics.record_operation(self.worker_regex.ar_in_qsize, inbound_q)
            return

        match = re.search(self.worker_regex.regex_ar_barrier_wait, log_entry)
        if match:
            duration = float(match.group(1))
            self.metrics.time_operation(self.worker_regex.ar_barrier_wait, duration)
            return

        match = re.search(self.worker_regex.regex_ar_queue_stats, log_entry)
        if match:
            inbound_max = int(match.group(1))
            inbound_mean = float(match.group(2))
            self.metrics.record_operation(self.worker_regex.ar_in_qsize_max, inbound_max)
            self.metrics.record_operation(self.worker_regex.ar_in_qsize_mean, inbound_mean)
            return

        match = re.search(self.worker_regex.regex_ar_chunk_gap_stats, log_entry)
        if match:
            self.metrics.record_operation(self.worker_regex.ar_chunk_gap_max, float(match.group(1)))
            self.metrics.record_operation(self.worker_regex.ar_chunk_gap_mean, float(match.group(2)))
            return

        match = re.search(self.worker_regex.regex_end_state_ar, log_entry)
        if match:
            if self.state_ar_begin_time is not None:
                diff_ms = calc_time_diff(self.state_ar_begin_time, match.group(1))
                self.metrics.time_operation(self.worker_regex.state_ar_time, diff_ms)
            self.state_ar_begin_time = None
            return

        # Start a stateful recording of time it takes to do all-reduce
        # The ordering of these two patterns should not change it's important for getting it correct
        # i.e. "Beginning optimizer step" should come before "Transitioning to epoch"
        match = re.search(self.worker_regex.regex_start_ar, log_entry)
        if match:
            self.begin_time = match.group(1)
            self.begin_epoch = int(match.group(2))
            self.metrics.record_operation(self.worker_regex.step, self.begin_epoch)

            if self.trans_time is None:
                return

            if self.begin_epoch == self.trans_epoch:
                diff_ms = calc_time_diff(self.trans_time, self.begin_time)
                self.metrics.time_operation(self.worker_regex.grad_accum_time, diff_ms)

        match = re.search(self.worker_regex.regex_end_ar, log_entry)
        if match:
            self.trans_time = match.group(1)
            self.trans_epoch = int(match.group(2))

            # haven't recorded start time yet
            if self.begin_time is None or self.begin_epoch is None:
                return

            if (self.trans_epoch - self.begin_epoch) != 1:
                self.metrics.record_error(self.worker_regex.missed_ar_step)
                # Error occured. Reset states
                self.begin_time = None
                self.begin_epoch = None
                self.trans_time = None
                self.trans_epoch = None
                return

            if (self.trans_epoch - self.begin_epoch) == 1:
                diff_ms = calc_time_diff(self.begin_time, self.trans_time)
                self.metrics.time_operation(self.worker_regex.ar_time, diff_ms)

    def run(self):
        while not self._stop_event.is_set():
            try:
                log_entry = self._log_queue.get(timeout=1).getMessage()
                if not self.start_logs:
                    continue
                self.process_log_entry(log_entry)
            except queue.Empty:
                continue


def calc_time_diff(time1_str: str, time2_str: str):
    """Find difference between two timestamps in milliseconds."""
    time_format = "%Y-%m-%d %H:%M:%S,%f"
    time1 = datetime.strptime(time1_str, time_format)
    time2 = datetime.strptime(time2_str, time_format)

    # difference in milliseconds
    return (time2 - time1).total_seconds() * 1000


def dump_unknown_error(log_string: str, dir_path: Path) -> bool:
    """Save unknown error log to a local file with timestamped filename."""
    pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    match = re.search(pattern, log_string)
    if match:
        datetime_str = match.group(1)
        datetime_str = datetime_str.replace(" ", "_").replace(":", "-").replace(",", "_")
        filename = dir_path / f"error_unknown_{datetime_str}.txt"
        # Write the entire string to file
        with open(filename, "w") as f:
            f.write(log_string)
        return True
    else:
        return False
