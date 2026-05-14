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
import contextlib

from enum import Enum
from itertools import chain
from typing import Any, Iterable, Sequence  # noqa: UP035

import torch

from agora_server.core.averaging.gradient_averager import GradientAverager

from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.dht import DHT
from hivemind.p2p import PeerID
from hivemind.utils import get_logger
from hivemind.utils.asyncio import enter_asynchronously
from hivemind.utils.math import get_flatten_greedy_dims


GatheredData = Any
logger = get_logger(__name__)


def qr_orthogonalize(matrix: torch.Tensor, iters: int = 1):
    """QR orthogonalization in-place for 2D matrix."""
    for _ in range(iters):
        Q, _ = torch.linalg.qr(matrix)
        matrix.copy_(Q)
    return matrix


def clip_tensor_norm_(tensors: Sequence[torch.Tensor] | torch.Tensor, max_norm: torch.Tensor, norm_type: float = 2.0):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    total_norm = torch.norm(torch.stack([torch.norm(t.detach(), norm_type) for t in tensors]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for t in tensors:
        t.mul_(clip_coef_clamped)

    return total_norm


class AllReducePhases(Enum):
    PHASE_P = 1
    PHASE_Q = 2


class PowerSGDGradientAverager(GradientAverager):
    """A gradient averager that implements PowerSGD compression: https://arxiv.org/abs/1905.13727;

    For basic properties and guaranties of gradient averagers, please refer to the base class docstring.
    Put simply, this method approximates large gradient tensors (m,n) with a product of two
    smaller matrices (m,r) by (r,n), where r is a parameter chosen by the user (see averager_rank).

    As a result, PowerSGD only needs to aggregate O((m + n) * r) tensors instead of O(m * n).
    High r, e.g. sqrt(max(m, n)) typically reduce communication by 2-8x without affecting convergence.
    Low r, e.g. 1-8, further accelerate communication, but may converge worse depending on the task.

    To maintain convergence with low r, this averager uses the error feedback strategy. Put simply,
    if some part of the gradient is "lost in compression", it will be added to the next iteration.
    This has two implications: (a) it needs more RAM in order to store the "feedback buffers"
    and (b) if devices stay alive only for one step, training with small rank may converge slower.
    This is because error feedback takes multiple steps to kick in.

    Since not all gradients are matrices, PowerSGD views 3d+ tensors via tensor.flatten(1, -1).
    If a tensor has less than 2 dimensions or does not compress efficiently, it will be aggregated
    normally, i.e. without powerSGD. See min_compression_ratio for details.

    **Note**: due to the above rule, PowerSGD is *not* shape-invariant. For instance, a
    matrix of shape (256, 256) be compressed differently if you .reshape it to (32, 32, 32).
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        averager_rank: int,
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: torch.device | None = None,
        client_mode: bool | None = None,
        warn: bool = True,
        min_compression_ratio: float = 0.5,
        averaged_grads: Sequence[torch.Tensor] | None = None,
        reset_buffers_every_k_steps: int = 10,
        init_error_norm: float = 10.0,
        error_norm_update_step: int = 10,
        **kwargs,
    ):
        """Initialize PowerSGD gradient averager.

        Args:
            parameters (Iterable[torch.nn.Parameter]): Pytorch parameters for which to aggregate gradients.
            averager_rank (int): Rank of compressed gradients.
            dht (DHT): A DHT instance connected to the rest of the swarm. See hivemind.DHT docs.
            prefix (str): A unique DHT key used for matchmaking. E.g. this can be your experiment name with optional suffixes.
            reuse_grad_buffers (bool, optional): If True, use model's .grad buffers for accumulating gradients over multiple steps. This is more memory efficient, but it requires that the user does *not* call zero_grad or clip_by_whatever at all. Defaults to False.
            accumulate_grads_on (torch.device | None, optional): If specified, accumulate gradients on this device. By default, this will use the same device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at the cost of extra time per step. If reuse_grad_buffers is True, this parameter has no effect. Defaults to None.
            client_mode (bool | None, optional): If False, this averager will accept incoming requests from other peers. If True, the averager will only join existing groups where at least one peer has client_mode=False. By default, this flag is copied from DHTNode inside the ``dht`` instance. Defaults to None.
            warn (bool, optional): If True, warn when the averager did not reset accumulators after use or did not use averaging results. Defaults to True.
            min_compression_ratio (float, optional): Apply PowerSGD to a tensor only if it reduces communication by at least this factor, otherwise aggregate tensors as is. Defaults to 0.5.
            averaged_grads (Sequence[torch.Tensor] | None, optional): If provided, it will be used as a set of averagable gradients. Defaults to None.
            reset_buffers_every_k_steps (int, optional): Reset compression buffers every this many steps. Defaults to 10.
            init_error_norm (float, optional): Initial error norm for compression. Defaults to 10.0.
            error_norm_update_step (int, optional): Update error norm every this many steps. Defaults to 10.
            **kwargs: Additional keyword arguments forwarded to the base GradientAverager.
        """
        self.rank = averager_rank
        self.parameters = tuple(parameters)
        self._uncompressed_gradients_indexes = set(
            i
            for i, grad in enumerate(self._grads_from_parameters())
            if grad.ndim <= 1
            or (1 - self.rank * sum(get_flatten_greedy_dims(grad)) / grad.numel()) < min_compression_ratio
            # compute how much parameters are left after factorization
        )
        self._ms = [
            torch.zeros_like(grad, device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients_indexes
        ]

        self._ms_copy = [
            torch.zeros_like(grad, device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients_indexes
        ]

        self._qs = [
            torch.rand((get_flatten_greedy_dims(grad)[1], self.rank), device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients_indexes
        ]

        # Error clipping warm-up
        self._init_error_norm = torch.zeros(1, device="cpu").share_memory_()
        self._init_error_norm.fill_(init_error_norm)
        self._error_norm_update_step = error_norm_update_step

        # Buffer reset tracking
        self.reset_buffers_every_k_steps = reset_buffers_every_k_steps
        self._step_count = 0
        self._last_successful_reset_step = 0

        super().__init__(
            self.parameters,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            averaged_grads=averaged_grads,
            **kwargs,
        )

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """Register a given group for one or more all-reduce rounds."""
        try:
            for phase in list(AllReducePhases):
                self._running_groups[group_info.group_id + phase.name.encode()] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            for phase in list(AllReducePhases):
                maybe_future = self._running_groups.pop(group_info.group_id + phase.name.encode(), None)
                if maybe_future and not maybe_future.done():
                    logger.warning(f"All-reduce group {group_info.group_id + phase.name.encode()} did not finish.")
            self._pending_groups_registered.clear()

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata."""
        self._step_count += 1  # Increment buffer step count
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as averaged_grads:
                averaged_grads_via_sgd = [
                    grad for idx, grad in enumerate(averaged_grads) if idx not in self._uncompressed_gradients_indexes
                ]

                err_norm = torch.nn.utils.get_total_norm(self._ms).item()
                logger.info(f"Error norm val: {err_norm:.6f}")

                prepsgd_norm = torch.nn.utils.get_total_norm(averaged_grads_via_sgd).item()
                logger.info(f"Prepsgd norm val: {prepsgd_norm:.6f}")

                # Adding noise to qs to prevent slow-down issues
                for q in self._qs:
                    q.add_(torch.randn_like(q) * 1e-30)

                # Make a copy of _ms in case of fail
                for m, ms_copy in zip(self._ms, self._ms_copy):
                    m.copy_(ms_copy)

                for grad, m in zip(averaged_grads_via_sgd, self._ms):
                    m.add_(grad.to(m.device))

                ps = [
                    torch.zeros((get_flatten_greedy_dims(grad)[0], self.rank), device="cpu")
                    for idx, grad in enumerate(averaged_grads_via_sgd)
                ]
                for p, q, m in zip(ps, self._qs, self._ms):
                    # we use reshape for all matrixes because PowerSGD works only with 2d tensors
                    torch.matmul(m.reshape(-1, q.size(0)), q, out=p)

                p_group_id = group_info.group_id + AllReducePhases.PHASE_P.name.encode()
                q_groud_id = group_info.group_id + AllReducePhases.PHASE_Q.name.encode()

                await self._run_allreduce_inplace_(ps, group_info, p_group_id, peer_fractions=peer_fractions, **kwargs)

                for p in ps:
                    p = qr_orthogonalize(p, iters=1)

                for p, q, m in zip(ps, self._qs, self._ms):
                    torch.matmul(m.reshape(-1, q.size(0)).t(), p, out=q)

                # local error before allreduce on Q
                for p, q, m in zip(ps, self._qs, self._ms):
                    new_m = torch.matmul(p, q.t()).reshape(m.size())
                    m.sub_(new_m)  # prev_err + grad - new_approx

                # Use the warmup error norm for first steps for averaging
                if self._step_count < self._error_norm_update_step:
                    local_err_norm = self._init_error_norm.clone()
                else:
                    local_err_norm = torch.nn.utils.get_total_norm(self._ms)

                phase_q_tensors = (
                    self._qs
                    + [grad for idx, grad in enumerate(averaged_grads) if idx in self._uncompressed_gradients_indexes]
                    + [local_err_norm]
                )

                await self._run_allreduce_inplace_(
                    phase_q_tensors, group_info, q_groud_id, peer_fractions=peer_fractions, **kwargs
                )

                average_err_norm = phase_q_tensors[-1]
                error_pre_clipped = clip_tensor_norm_(self._ms, average_err_norm)
                logger.info(f"Error pre-clip norm val: {error_pre_clipped:.6f}")

                for p, q, ms_copy, grad, m in zip(ps, self._qs, self._ms_copy, averaged_grads_via_sgd, self._ms):
                    new_m = torch.matmul(p, q.t()).reshape(ms_copy.size())
                    grad.copy_(new_m)
                    ms_copy.copy_(m)

                postpsgd_norm = torch.nn.utils.get_total_norm(averaged_grads_via_sgd).item()
                logger.info(f"Postpsgd norm val: {postpsgd_norm:.6f}")

                return user_gathered
        except BaseException as e:
            logger.error(e, exc_info=True)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    def get_current_state(self):
        """Get current gradient averager state and when requested by a newbie peer."""
        with torch.no_grad(), self.lock_averaged_tensors:
            grad_averager_buffers = [q for q in self._qs]
            grad_averager_buffers_infos = [
                CompressionInfo.from_tensor(buffer, key=f"buffer_q_{key}", role=TensorRole.GRADIENT)
                for buffer, key in zip(grad_averager_buffers, enumerate(grad_averager_buffers))
            ]

            error_buffers = [ms for ms in self._ms_copy]
            error_buffers_infos = [
                CompressionInfo.from_tensor(buffer, key=f"buffer_ms_{key}", role=TensorRole.GRADIENT)
                for buffer, key in zip(error_buffers, enumerate(error_buffers))
            ]

        metadata = dict(group_bits=self.get_group_bits())

        all_tensors = list(chain(grad_averager_buffers, error_buffers))
        all_tensor_infos = list(chain(grad_averager_buffers_infos, error_buffers_infos))

        return metadata, all_tensors, all_tensor_infos

    def load_state_from_peers(self, peer_id: PeerID | None = None, **kwargs):
        """Attempt to download the latest optimizer state from peers and update gradient averager buffers."""
        if peer_id is not None:
            kwargs["target_peer"] = peer_id
        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors, _ = loaded_state
        logger.info("Starting loading gradient averager buffers from peers")

        if len(flat_tensors) != len(self._qs) + len(self._ms_copy):
            logger.error("Failed to load state from peer, received invalid parameters, extras or metadata")
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            for local_q, loaded_q in zip(self._qs, flat_tensors[: len(self._qs)]):
                local_q.copy_(loaded_q, non_blocking=True)
            for local_ms, local_ms_copy, loaded_ms in zip(self._ms, self._ms_copy, flat_tensors[len(self._qs) :]):
                local_ms.copy_(loaded_ms, non_blocking=True)
                local_ms_copy.copy_(local_ms_copy, non_blocking=True)

        # Initialize the error norm to use for first steps
        init_error_norm = torch.norm(torch.stack([torch.norm(t.detach()) for t in local_ms])).item()
        self._init_error_norm.fill_(init_error_norm)
