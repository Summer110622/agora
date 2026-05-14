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

import asyncio

from typing import Any, Iterable, Sequence  # noqa: UP035

import torch

from agora_server.core.averaging.gradient_averager import GradientAverager

from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.dht import DHT
from hivemind.utils import get_logger
from hivemind.utils.asyncio import enter_asynchronously


GatheredData = Any
logger = get_logger(__name__)


class RandomProjGradientAverager(GradientAverager):
    """A gradient averager that uses random projection compression.

    This class implements gradient averaging by using random matrix projection.

    The compression is applied only to gradients with more than 1 dimension (matrices), while
    1D gradients (biases, layer norms) are sent uncompressed.

    **Note**:
        - Compression is only applied to gradients with ndim > 1 (weight matrices).
        - 1D gradients (biases, layer norm parameters) are sent uncompressed.
        - Random projections are refreshed periodically and in sync with other peers in a stage.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        grad_compression_factor: int,
        refresh_projection_rate: int = 5,
        grad_warmup_steps: int = 0,
        discount_grad: bool = True,
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: torch.device | None = None,
        client_mode: bool | None = None,
        warn: bool = True,
        averaged_grads: Sequence[torch.Tensor] | None = None,
        **kwargs,
    ):
        """Initialize gradient averager with compression using random projection.

        Args:
            parameters (Iterable[torch.nn.Parameter]): Pytorch parameters for which to aggregate gradients.
            grad_compression_factor (int): Compression factor applied to gradients. The actual compression is split evenly between matrix projection (factor/2) and column sampling (factor/2).
            refresh_projection_rate (int, optional): How often to refresh the random projection matrices and column indices, measured in epochs. Defaults to 5.
            grad_warmup_steps (int, optional): Number of warmup steps before applying compression. Defaults to 0.
            discount_grad (bool, optional): Whether to apply gradient discounting. Defaults to True.
            dht (DHT): A DHT instance connected to the rest of the swarm. See hivemind.DHT docs.
            prefix (str): A unique DHT key used for matchmaking. E.g. this can be your experiment name with optional suffixes.
            reuse_grad_buffers (bool, optional): If True, use model's .grad buffers for accumulating gradients over multiple steps. This is more memory efficient, but it requires that the user does *not* call zero_grad or clip_by_whatever at all. Defaults to False.
            accumulate_grads_on (torch.device | None, optional): If specified, accumulate gradients on this device. By default, this will use the same device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at the cost of extra time per step. If reuse_grad_buffers is True, this parameter has no effect. Defaults to None.
            client_mode (bool | None, optional): If False, this averager will accept incoming requests from other peers. If True, the averager will only join existing groups where at least one peer has client_mode=False. By default, this flag is copied from DHTNode inside the ``dht`` instance. Defaults to None.
            warn (bool, optional): If True, warn when the averager did not reset accumulators after use or did not use averaging results. Defaults to True.
            averaged_grads (Sequence[torch.Tensor] | None, optional): If provided, it will be used as a set of averagable gradients. Defaults to None.
            **kwargs: Additional arguments passed to the parent GradientAverager, including request_timeout and min_matchmaking_time.
        """
        self.refresh_projection_rate = refresh_projection_rate
        self.compression_factor_mat = grad_compression_factor
        self.compression_factor_col = grad_compression_factor
        self.grad_warmup_steps = grad_warmup_steps
        self.discount_grad = discount_grad
        self.parameters = tuple(parameters)
        self.processed_batches = torch.tensor(0.0, device="cpu").share_memory_()

        self.idx_no_require_grad = set(i for i, param in enumerate(parameters) if not param.requires_grad)
        self._uncompressed_gradients_indexes = set(
            i for i, grad in enumerate(self._grads_from_parameters()) if grad.ndim <= 1
        )
        self._uncompressed_gradients_indexes = self._uncompressed_gradients_indexes.difference(
            self.idx_no_require_grad
        )

        stage = prefix.split("_")[0]
        self.do_compression = False if stage in ["head", "tail"] else True

        # Random matrix
        self.last_seed = torch.tensor(0, device="cpu").share_memory_()
        self._rs = self.get_random_projection(self.last_seed)

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

    def get_random_projection(self, seed):
        rs = []
        torch.manual_seed(seed)
        for idx, grad in enumerate(self._grads_from_parameters()):
            if idx not in (self._uncompressed_gradients_indexes.union(self.idx_no_require_grad)):
                mat_shape = grad.shape
                r = torch.randn(
                    mat_shape[1], int(mat_shape[1] / self.compression_factor_mat), device="cpu"
                ).share_memory_()
                r_orth, _ = torch.linalg.qr(r)
                rs.append(r_orth)
        return rs

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run approximate PowerSGD aggregation with a single communication phase."""
        try:
            # Deserialize metadata from peers
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # Compute peer fractions for load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as averaged_grads:
                epoch = self.global_epoch
                snapped_epoch = int(round(epoch / self.refresh_projection_rate) * self.refresh_projection_rate)

                # Determine whether to use compression
                use_compression = snapped_epoch > self.grad_warmup_steps and self.do_compression

                # Separate gradients if using compression
                if use_compression:
                    compressed_grads = [
                        grad
                        for i, grad in enumerate(averaged_grads)
                        if i not in self._uncompressed_gradients_indexes.union(self.idx_no_require_grad)
                    ]
                    uncompressed_grads = [
                        grad for i, grad in enumerate(averaged_grads) if i in self._uncompressed_gradients_indexes
                    ]

                    # Apply random projections to compress gradients
                    g_compressed = []
                    for g, r in zip(compressed_grads, self._rs):
                        g_c = torch.matmul(g, r).contiguous()
                        g_compressed.append(g_c)

                    all_tensors = g_compressed + [g.contiguous() for g in uncompressed_grads]

                    if self.discount_grad:
                        # Multiply by batches processed
                        scaled_tensors = [self.processed_batches * t for t in all_tensors]
                        all_tensors = [self.processed_batches] + scaled_tensors

                        # All-reduce
                        await self._run_allreduce_inplace_(
                            all_tensors, group_info, peer_fractions=peer_fractions, **kwargs
                        )

                        mean_batches = all_tensors[0]
                        g_compressed_avg = all_tensors[1 : 1 + len(g_compressed)]
                        g_uncompressed_avg = all_tensors[1 + len(g_compressed) :]

                        total_batches = mean_batches * group_info.group_size
                        num_peers = len(user_gathered)
                        factor = num_peers / total_batches if total_batches > 1 else 0

                        # Reconstruct and apply discounted gradients
                        for gc, r, grad in zip(g_compressed_avg, self._rs, compressed_grads):
                            new_gc = torch.matmul(gc, r.T)
                            grad.copy_(factor * new_gc)

                        for guc, grad in zip(g_uncompressed_avg, uncompressed_grads):
                            grad.copy_(factor * guc)

                    else:
                        await self._run_allreduce_inplace_(
                            all_tensors, group_info, peer_fractions=peer_fractions, **kwargs
                        )
                        g_compressed_avg = all_tensors[: len(g_compressed)]

                        # Reconstruct full gradients
                        for gc, r, grad in zip(g_compressed_avg, self._rs, compressed_grads):
                            new_gc = torch.matmul(gc, r.T)
                            grad.copy_(new_gc)

                else:
                    # No compression
                    if self.discount_grad:
                        gradient_idxs = [i for i in range(len(averaged_grads)) if i not in self.idx_no_require_grad]
                        scaled_grads = [self.processed_batches * averaged_grads[i] for i in gradient_idxs]
                        all_tensors = [self.processed_batches] + scaled_grads

                        await self._run_allreduce_inplace_(
                            all_tensors, group_info, peer_fractions=peer_fractions, **kwargs
                        )

                        mean_batches = all_tensors[0]
                        g_avg = all_tensors[1:]
                        total_batches = mean_batches * group_info.group_size
                        num_peers = len(user_gathered)
                        factor = num_peers / total_batches if total_batches > 1 else 0

                        for i, grad_avg in zip(gradient_idxs, g_avg):
                            averaged_grads[i].copy_(factor * grad_avg)
                    else:
                        await self._run_allreduce_inplace_(
                            averaged_grads, group_info, peer_fractions=peer_fractions, **kwargs
                        )

                return user_gathered

        except BaseException as e:
            if isinstance(e, Exception):
                logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")
