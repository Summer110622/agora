# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2026
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import multiprocessing as mp
import random
import threading

from collections.abc import Sequence
from functools import partial
from time import perf_counter
from typing import Any, Callable

import torch

from agora_server.core.averaging.gradient_averager import GradientAveragerFactory
from agora_server.core.averaging.state_averager import TrainingStateAveragerFactory
from agora_server.core.optimization.optimizer_sync import Optimizer
from agora_server.core.server.dht_handler import DHTHandlerThread, get_experts
from agora_server.core.server.module_collab import ModuleCollab, _get_autocast_context
from agora_server.core.server.runtime import Runtime
from agora_server.logging.log_monitor import LogMonitor
from agora_server.models.base_arguments import ModelArguments
from agora_server.models.expert_registry import ExpertRegistry
from agora_server.models.lr_schedule import schedule_name_to_scheduler
from agora_server.monitor.dht_monitor import patch_dht_protocol_logging
from agora_server.monitor.peer_visibility import PeerVisibilityMonitor
from agora_server.types import ServerCreationError, TorchOptimizer, TorchOptimizerFactory
from agora_server.utils.subspace import load_ss_components

from hivemind.dht import DHT
from hivemind.moe.expert_uid import UID_DELIMITER
from hivemind.moe.server.connection_handler import ConnectionHandler as _BaseConnectionHandler
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.asyncio import switch_to_uvloop
from hivemind.utils.logging import get_logger
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor


logger = get_logger(__name__)


class ConnectionHandler(_BaseConnectionHandler):
    """ConnectionHandler that properly cleans up P2P client sockets on shutdown."""

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()
        stop = asyncio.Event()
        loop.add_reader(self._inner_pipe.fileno(), stop.set)

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p, balanced=self.balanced)
                self.ready.set_result(None)
            except Exception as e:
                logger.error("ConnectionHandler failed to start:", exc_info=True)
                self.ready.set_exception(e)
                return

            try:
                await stop.wait()
            finally:
                await self.remove_p2p_handlers(self._p2p)
                if self._p2p is not None:
                    await self._p2p.shutdown()

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")


def _compile_warmup(
    module: torch.nn.Module,
    sample_input_func: Callable,
    input_schema: dict,
    device: torch.device | str,
    use_mixed_precision: bool,
    expert_uid: str,
    num_warmup_batches: int = 1,
) -> None:
    """Run dummy forward+backward passes to trigger torch.compile before the worker starts serving.

    This ensures every worker is already compiled when it joins the network, avoiding the cascading
    starvation problem where the first-compiled worker hogs all traffic from the heap-based balancer.
    """
    device = torch.device(device) if isinstance(device, str) else device

    for i in range(num_warmup_batches):
        t_start = perf_counter()

        # Generate dummy inputs matching the real serving schema
        # Zero out float tensors because sample_input_func may use torch.empty (uninitialized memory),
        # which can contain NaN/Inf and cause CUDA asserts in loss functions like cross_entropy.
        sample_input = sample_input_func(DUMMY_BATCH_SIZE, **input_schema)
        if not isinstance(sample_input, (list, tuple)):
            sample_input = (sample_input,)
        inputs = []
        for t in sample_input:
            if isinstance(t, torch.Tensor):
                t = t.to(device)
                if t.is_floating_point():
                    t.zero_()
                inputs.append(t)
            else:
                inputs.append(t)

        # Forward warmup (matches ModuleBackend.forward / ModuleCollab.forward -- no_grad + autocast)
        t_fwd_start = perf_counter()
        with torch.no_grad():
            with _get_autocast_context(use_mixed_precision, device):
                outputs = module(*inputs)
        t_fwd = perf_counter() - t_fwd_start

        # Backward warmup (matches ModuleCollab.backward -- enable_grad + autocast on forward only)
        t_bwd_start = perf_counter()
        with torch.enable_grad():
            inputs_grad = [t.detach().requires_grad_(True) if t.is_floating_point() else t.detach() for t in inputs]
            with _get_autocast_context(use_mixed_precision, device):
                outputs = module(*inputs_grad)

            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)
            outputs_flat = tuple(o for o in outputs if isinstance(o, torch.Tensor) and o.requires_grad)
            grad_tensors = [torch.ones_like(o) for o in outputs_flat]
            torch.autograd.backward(outputs_flat, grad_tensors=grad_tensors, create_graph=False, retain_graph=False)
        t_bwd = perf_counter() - t_bwd_start

        # Clean up -- no optimizer step, so weights are unchanged.
        # Use set_to_none=False to preserve grad buffer identity for torch.compile:
        # compiled backward graphs hold references to the original grad tensors,
        # and setting grad = None would cause them to write to freed/reused memory.
        module.zero_grad(set_to_none=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t_total = perf_counter() - t_start
        logger.info(
            f"torch.compile warmup batch {i + 1}/{num_warmup_batches} for {expert_uid}: "
            f"forward={t_fwd:.2f}s, backward={t_bwd:.2f}s, total={t_total:.2f}s"
        )


class Server(threading.Thread):
    """Server allows you to host "experts" - pytorch subnetworks that can be accessed remotely by peers.

    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does two things:
        - processes incoming forward/backward requests via Runtime (created by the server)
        - publishes updates to expert status every :update_period: seconds
    """

    def __init__(
        self,
        dht: DHT,
        module_backends: dict[str, ModuleBackend],
        optim_collab: Optimizer,
        checkpoint_saver: Any | None = None,
        num_connection_handlers: int = 1,
        update_period: float = 30,
        expiration: float | None = None,
        start: bool = False,
        **kwargs,
    ):
        """Initialize Server.

        Args:
            dht (DHT): An instance of hivemind.DHT. Server will use DHT for all network interactions.
            module_backends (dict[str, ModuleBackend]): A dict {expert uid (str) : ModuleBackend} for all experts hosted by this server.
            optim_collab (Optimizer): An instance of collaborative optimizer.
            checkpoint_saver (Any | None, optional): Checkpoint saver instance. Defaults to None.
            num_connection_handlers (int, optional): Maximum number of simultaneous requests. Please note that the default value of 1 is too small for normal functioning, we recommend 4 handlers per expert backend. Defaults to 1.
            update_period (float, optional): How often will server attempt to publish its state (i.e. experts) to the DHT; if dht is None, this parameter is ignored. Defaults to 30.
            expiration (float | None, optional): When server declares its experts to the DHT, these entries will expire after this many seconds. Defaults to None.
            start (bool, optional): If True, the server will immediately start as a background thread and returns control after server is ready (see .ready below). Defaults to False.
            checkpoint_dir (Path | None, optional): Directory to save checkpoints. Defaults to None.
            **kwargs: Additional keyword arguments passed to Runtime.
        """
        super().__init__()
        self.dht, self.module_backends, self.update_period = dht, module_backends, update_period
        self.optim_collab = optim_collab
        self.checkpoint_saver = checkpoint_saver

        self.conn_handlers = [ConnectionHandler(dht, self.module_backends) for _ in range(num_connection_handlers)]
        self.runtime = Runtime(self.module_backends, **kwargs)

        if self.module_backends:
            self.dht_handler_thread = DHTHandlerThread(
                module_backends=self.module_backends,
                dht=self.dht,
                update_period=self.update_period,
                expiration=expiration,
                daemon=True,
            )

        if start:
            self.run_in_background(await_ready=True)

    @classmethod
    def create(
        cls,
        model_name: str,
        model_conf: ModelArguments,
        expert_name: str,
        collaborative_optim_factory: Callable[..., Optimizer],
        torch_optim_factory: TorchOptimizer | TorchOptimizerFactory,
        weight_decay: float,
        bandwidth: float | None,
        state_avg_factory: TrainingStateAveragerFactory,
        grad_avg_factory: GradientAveragerFactory | None,
        scheduler: str,
        num_warmup_steps: int | None,
        num_training_steps: int | None,
        min_batch_size: int,
        max_batch_size: int,
        stats_report_interval: int,
        coalesce_batches: bool = True,
        min_lr_ratio: float = 0.1,
        update_period: float = 30,
        num_experts: int | None = None,
        expert_uids: list[str] | None = None,
        expert_pattern: str | None = None,
        no_optim_params: list[str] | None = None,
        no_decay_params: list[str] | None = None,
        num_handlers: int | None = None,
        seed_peer: bool = False,
        device: torch.device | str | None = None,
        enable_tf32: bool = False,
        use_mixed_precision: bool = False,
        initial_peers: Sequence[str] = (),
        dump_addrs: str | None = None,
        compression: CompressionType = CompressionType.NONE,
        custom_module_path: str | None = None,
        expiration: float | None = None,
        aws_access: dict | None = None,
        checkpoint_manager_factory: Callable[..., Any] | None = None,  # noqa: E501 using Any type here for now until we create an abstract checkpoint saver class
        load_checkpoint: bool = False,
        prom_monitor_callback: Callable[..., Any] | None = None,
        use_peer_visibility_monitor: bool = False,
        use_dht_monitor: bool = False,
        log_monitor: LogMonitor | None = None,
        delay_range_forward_backward: tuple[float, float] = (0.0, 0.0),
        use_torch_compile: bool = False,
        compile_warmup_batches: int = 1,
        start: bool = False,
        **kwargs,
    ) -> "Server":
        """Instantiate a server similar to hivemind moe server but support collaborative optimisation.

        Args:
            model_name (str): The model name, e.g. "llama", "mnist", etc.
            model_conf (ModelArguments): Model configuration.
            expert_name (str): Expert type e.g. "lm_head", "lm_body", etc.
            collaborative_optim_factory (Callable[..., Optimizer]): Factory for creating collaborative optimizer.
            torch_optimizer_factory: TorchOptimizer | TorchOptimizerFactory: Use this torch optimizer for training.
            weight_decay (float): Weight decay value for torch optimizer.
            bandwidth (float | None): If specified, this value represents the network bandwidth available to averager.
            state_avg_factory (TrainingStateAveragerFactory): Factory for creating state averager.
            grad_avg_factory (GradientAveragerFactory | None): Factory for creating gradient averager. If None, no gradient averaging is performed.
            scheduler (str): If not `none`, the name of the expert LR scheduler.
            num_warmup_steps (int | None): The number of warmup steps for LR schedule.
            num_training_steps (int | None): The total number of steps for LR schedule.
            min_lr_ratio (float, optional): The minimum learning rate as a ratio of the initial learning rate. Defaults to 0.1.
            min_batch_size (int): Total num examples in the same batch will be greater than this value.
            max_batch_size (int): Total num examples in the same batch will not exceed this value.
            stats_report_interval (int): Interval between two reports of batch processing performance statistics.
            update_period (float, optional): Period for updating DHT. Defaults to 30.
            num_experts (int | None, optional): Run this many identical experts. Defaults to None.
            expert_uids (list[str] | None, optional): Spawn experts with these exact uids, overrides num_experts and expert_pattern. Defaults to None.
            expert_pattern (str | None, optional): A string pattern or a list of expert uids, example: myprefix.[0:32].[0:256] means "sample random experts between myprefix.0.0 and myprefix.255.255". Defaults to None.
            no_optim_params (list[str] | None, optional): List of parameter name substrings that should not be passed to optimizer. Defaults to None.
            no_decay_params (list[str] | None, optional): List of parameter name substrings that should not be weight-decayed. Defaults to None.
            num_handlers (int | None, optional): Server will use this many parallel processes to handle incoming requests. Defaults to None.
            seed_peer (bool, optional): Whether this is a seed peer. Defaults to False.
            device (torch.device | str | None, optional): All experts will use this device in torch notation; default: cuda if available else cpu. Defaults to None.
            enable_tf32 (bool, optional): Whether to enable TF32 precision on supported devices (see docs/mixed_precision.md). Defaults to False.
            use_mixed_precision (bool, optional): Whether to enable BF16 mixed precision (see docs/mixed_precision.md). Defaults to False.
            initial_peers (Sequence[str], optional): Multiaddrs of one or more active DHT peers (if you want to join an existing DHT). Defaults to ().
            dump_addrs (str | None, optional): Path to dump addresses. Defaults to None.
            compression (CompressionType, optional): If specified, use this compression to pack all inputs, outputs and gradients by all experts hosted on this server. For a more fine-grained compression, start server in python and specify compression for each BatchTensorProto in ModuleBackend for the respective experts. Defaults to CompressionType.NONE.
            custom_module_path (str | None, optional): Path to custom model experts. Defaults to None.
            expiration (float | None, optional): Expiration time for DHT records. Defaults to None.
            aws_access (dict | None, optional): AWS access credentials. Defaults to None.
            checkpoint_manager_factory (Callable[..., Any] | None, optional): Factory for creating checkpoint saver or loader. Defaults to None.
            load_checkpoint (bool, optional): Whether to load from checkpoint. Defaults to False.
            prom_monitor_callback (Callable[..., Any] | None, optional): Prometheus monitoring callback. Defaults to None.
            use_peer_visibility_monitor (bool, optional): Whether to use peer visibility monitoring. Defaults to False.
            use_dht_monitor (bool, optional): Whether to patch DHT protocol logging for RPC monitoring. Defaults to False.
            log_monitor (LogMonitor | None, optional): Log monitor instance. Defaults to None.
            delay_range_forward_backward (tuple[float, float], optional): Delay range for forward/backward passes. Defaults to (0.0, 0.0).
            use_torch_compile (bool, optional): Whether to enable torch.compile for forward/backward call-sites. Defaults to False.
            compile_warmup_batches (int, optional): Number of warmup batches to run before serving. Defaults to 1.
            start (bool, optional): If True, starts server right away and returns when server is ready for requests. Defaults to False.
            **kwargs: Any other params will be forwarded to DHT upon creation.

        Returns:
            Server: The created server instance.

        Raises:
            ServerCreationError: If creation is interrupted by KeyboardInterrupt (SIGINT).
        """
        # Verify uid parameters
        if not (
            (expert_pattern is None and num_experts is None and expert_uids is not None)
            or (num_experts is not None and expert_uids is None)
        ):
            logger.error(
                "Please provide either expert_uids *or* num_experts (possibly with expert_pattern), but not both"
            )
            raise ValueError("Invalid expert uid parameters")

        # Get expert class and sample input function from registry
        if custom_module_path:
            ExpertRegistry.add_custom_models(custom_module_path)

        try:
            expert_class, sample_input_func = ExpertRegistry.get_expert_info(model=model_name, expert_name=expert_name)
        except Exception as e:
            logger.error(f"Failed to create expert {expert_name} from registry: {e}")
            raise

        dht = None
        experts: dict[str, ModuleCollab] = {}
        optim_collab = None

        try:
            # Connect to DHT
            if use_dht_monitor:
                _ = patch_dht_protocol_logging()
            dht = DHT(initial_peers=initial_peers, start=True, startup_timeout=30, **kwargs)
            visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
            logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

            # Connect to monitor
            if log_monitor:
                log_monitor.connect_dht(dht)

            if dump_addrs is not None:
                with open(dump_addrs, "w") as text_file:
                    text_file.write(visible_maddrs_str[-1])

            # Generate uids
            if expert_uids is None:
                expert_uids = []
                uids_to_generate = num_experts - len(expert_uids)
                if uids_to_generate > 0:
                    logger.info(f"Generating {uids_to_generate} expert uids from pattern {expert_pattern}")
                    expert_uids.extend(_generate_uids(uids_to_generate, expert_pattern, dht))

            # Add optional monitors
            if use_peer_visibility_monitor:
                stage = expert_uids[0].split(".")[0]
                stage_uids = stage + "."
                _ = PeerVisibilityMonitor(dht, [stage_uids])

            num_experts = len(expert_uids)
            num_handlers = num_handlers if num_handlers is not None else num_experts * 8
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # TF32 is significantly faster than FP32 on modern GPUs
            logger.info("Current matmul precision: " + torch.get_float32_matmul_precision())
            if enable_tf32 and device != "cpu":
                logger.info("Enabling TF32 matmul precision")
                torch.set_float32_matmul_precision("high")

            # BF16 mixed precision for higher throughput on modern GPUs (H100, L40S, etc.)
            if use_mixed_precision:
                if device == "cpu":
                    logger.warning("Mixed precision requested but device is CPU, disabling")
                    use_mixed_precision = False
                elif not torch.cuda.is_bf16_supported():
                    logger.warning("Mixed precision requested but BF16 not supported on this GPU, disabling")
                    use_mixed_precision = False
                else:
                    logger.info("BF16 mixed precision enabled for forward/backward passes")

            # Scheduler
            scheduler_cls = schedule_name_to_scheduler[scheduler]
            if scheduler_cls is not None:
                scheduler_cls = partial(
                    scheduler_cls,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                )

            # Initialize experts
            input_schema = model_conf.input_schema
            sample_input = sample_input_func(DUMMY_BATCH_SIZE, **input_schema)
            if isinstance(sample_input, Sequence):
                args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
            else:
                args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

            logger.info("Initializing expert")
            for expert_uid in expert_uids:
                expert = expert_class(model_conf)

                # Monitor callbacks
                if prom_monitor_callback:
                    prom_monitor_callback(
                        expert,
                        model_conf,
                        use_mixed_precision=use_mixed_precision,
                        enable_tf32=enable_tf32,
                    )

                # Select parameters for optimizer
                no_decay_params = no_decay_params or []
                no_optim_params = no_optim_params or []

                params = [
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if not any(nd in n for nd in no_decay_params)
                            and not any(no in n for no in no_optim_params)
                        ],
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if any(nd in n for nd in no_decay_params) and not any(no in n for no in no_optim_params)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                params = [group for group in params if group["params"]]

                # Load subspace components if needed
                if model_conf.use_compression and model_conf.ss_component_path:
                    if not aws_access:
                        aws_access = {}

                    ss_comps = load_ss_components(
                        model_conf.ss_component_path,
                        **aws_access,
                    )
                    expert.load_comp(ss_comps)
                    logger.info("Succeeded loading remote subspace components")
                    expert.ss_regularize()

                optimizer_lock = mp.Lock()

                backend = ModuleCollab(
                    optimizer_lock=optimizer_lock,
                    name=expert_uid,
                    module=expert,
                    args_schema=args_schema,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    delay_range_forward_backward=delay_range_forward_backward,
                    use_mixed_precision=use_mixed_precision,
                    use_torch_compile=use_torch_compile,
                )

                backend.module.to(device)

                if use_torch_compile and compile_warmup_batches > 0:
                    try:
                        _compile_warmup(
                            module=backend._compiled_module,
                            sample_input_func=sample_input_func,
                            input_schema=input_schema,
                            device=device,
                            use_mixed_precision=use_mixed_precision,
                            expert_uid=expert_uid,
                            num_warmup_batches=compile_warmup_batches,
                        )
                    except Exception as e:
                        logger.warning(
                            f"torch.compile warmup failed for {expert_uid}, falling back to lazy compilation: {e}"
                        )

                # TODO: Add Swarm LambWithGradientClipping from torch_optim
                optim_collab = collaborative_optim_factory(
                    model=expert,
                    optimizer_lock=optimizer_lock,
                    optimizer=torch_optim_factory,
                    params=params,
                    dht=dht,
                    run_id=expert_uid.split(UID_DELIMITER)[0],
                    scheduler=scheduler_cls,
                    state_avg_factory=state_avg_factory,
                    grad_averager_factory=grad_avg_factory,
                    bandwidth=bandwidth,
                )

                # Assign optimizer to backend immediately so cleanup can find it
                backend.optimizer = optim_collab
                experts[expert_uid] = backend

                if seed_peer:
                    logger.info("Seed node - not loading weights from peers")
                elif load_checkpoint:
                    logger.info("Will load state from checkpoint, skipping loading from peers")
                else:
                    logger.info("Loading weights from peers")
                    optim_collab.load_state_from_peers(wait_for_end_round=True)

                optim_collab.tracker.allow_progress_report = True
                optim_collab.set_averagers_allow_state_sharing()

            # Optionally create checkpoint saver and load experts
            try:
                checkpoint_saver = checkpoint_manager_factory(experts) if checkpoint_manager_factory else None
            except Exception as e:
                logger.error(f"Failed to create checkpoint saver: {e}")
                raise

            if load_checkpoint and checkpoint_saver:
                try:
                    checkpoint_saver.load_experts()
                except Exception as e:
                    logger.error(f"Failed to load experts from checkpoint: {e}")
                    raise

            optim_collab.wait_for_join_window()

            return cls(
                dht=dht,
                module_backends=experts,
                optim_collab=optim_collab,
                checkpoint_saver=checkpoint_saver,
                num_connection_handlers=num_handlers,
                update_period=update_period,
                expiration=expiration,
                start=start,
                device=device,
                stats_report_interval=stats_report_interval,
                coalesce_batches=coalesce_batches,
            )

        except KeyboardInterrupt:
            logger.info("Server creation interrupted, cleaning up partially initialized components...")
            if optim_collab is not None:
                logger.info("Shutting down optimizer")
                optim_collab.shutdown()
            if dht is not None:
                logger.info("Shutting down DHT")
                dht.shutdown()
            raise ServerCreationError("Server creation interrupted by shutdown signal") from None

    def run(self):
        """Start Server in the current thread.

        Initialize dht if necessary, start connection handlers, run Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Server started with {len(self.module_backends)} modules:")
        for expert_name, backend in self.module_backends.items():
            num_parameters = sum(p.numel() for p in backend.module.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.module.__class__.__name__}, {num_parameters} parameters")

        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        if self.module_backends:
            self.dht_handler_thread.start()

        if self.checkpoint_saver is not None and isinstance(self.checkpoint_saver, threading.Thread):
            self.checkpoint_saver.start()

        for handler in self.conn_handlers:
            handler.run_in_background()

        self.optim_collab.start_monitor()

        self.runtime.run()

    def run_in_background(self, await_ready: bool = True, timeout: float | None = None):
        """Start Server in a background thread.

        If `await_ready`, this method will wait until background server is ready to process incoming requests or for `timeout` seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """An event (multiprocessing.Event) that is set when the server is ready to process requests.

        **Example:**

        >>> server.start()
        >>> server.ready.wait(timeout=10)
        >>> print("Server ready" if server.ready.is_set() else "Server didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def shutdown(self):
        """Gracefully terminate the server, process-safe.

        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.ready.clear()

        self.optim_collab.shutdown()

        for handler in self.conn_handlers:
            handler.shutdown()
        logger.debug("Connection handlers terminated")

        if self.module_backends:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

        if self.checkpoint_saver is not None and isinstance(self.checkpoint_saver, threading.Thread):
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        self.dht.shutdown()

        logger.debug("Shutting down runtime")
        self.runtime.shutdown()

        logger.info("Server shutdown successfully")


def _generate_uids(
    num_experts: int,
    expert_pattern: str | None,
    dht: DHT | None = None,
    attempts_per_expert: int = 10,
) -> list[str]:
    """Sample experts from a given pattern, remove duplicates.

    Args:
        num_experts (int): Sample this many unique expert uids.
        expert_pattern (str | None): A string pattern or a list of expert uids, example: myprefix.[0:32].[0:256] means "sample random experts between myprefix.0.0 and myprefix.255.255".
        dht (DHT | None, optional): If specified, uses this DHT to check that expert uids are not yet occupied by other peers. Defaults to None.
        attempts_per_expert (int, optional): Give up if unable to generate a new expert uid after this many attempts per uid. Defaults to 10.

    Returns:
        list[str]: List of generated unique expert uids.

    **Note:**
        This method is not strictly process-safe. If several servers run it concurrently, they have a small chance of sampling duplicate expert uids.
    """
    remaining_attempts = attempts_per_expert * num_experts
    found_uids, attempted_uids = list(), set()

    def _generate_uid():
        if expert_pattern is None:
            return f"expert{UID_DELIMITER}{attempts_per_expert * num_experts - remaining_attempts}"

        uid = []
        for block in expert_pattern.split(UID_DELIMITER):
            try:
                if "[" not in block and "]" not in block:
                    uid.append(block)
                elif block.startswith("[") and block.endswith("]") and ":" in block:
                    slice_start, slice_end = map(int, block[1:-1].split(":"))
                    uid.append(str(random.randint(slice_start, slice_end - 1)))
                else:
                    raise ValueError("Block must be either fixed or a range [from:to]")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise ValueError(f"Expert pattern {expert_pattern} has invalid block {block}, {e}") from e
        return UID_DELIMITER.join(uid)

    while remaining_attempts > 0 and len(found_uids) < num_experts:
        # 1. sample new expert uids at random
        new_uids = []
        while len(new_uids) + len(found_uids) < num_experts and remaining_attempts > 0:
            new_uid = _generate_uid()
            remaining_attempts -= 1
            if new_uid not in attempted_uids:
                attempted_uids.add(new_uid)
                new_uids.append(new_uid)

        # 2. look into DHT (if given) and remove duplicates
        if dht is not None:
            existing_expert_uids = {
                found_expert.uid for found_expert in get_experts(dht, new_uids) if found_expert is not None
            }
            new_uids = [new_uid for new_uid in new_uids if new_uid not in existing_expert_uids]

        found_uids += new_uids

    if len(found_uids) != num_experts:
        logger.warning(
            f"Found only {len(found_uids)} out of {num_experts} free expert uids after "
            f"{attempts_per_expert * num_experts} attempts"
        )
    return found_uids
