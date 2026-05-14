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
import threading

from pathlib import Path

import requests

from agora.security.integrity_check import verify_integrity
from agora.utils.connection_test_server import TestServer
from agora.utils.state_loader import StateDownloader
from agora_server.security.authorizer import PluralisAuthorizer
from agora_server.utils.node_info import NodeInfo
from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from hivemind.utils.crypto import RSAPublicKey
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class NonRetriableError(Exception):
    pass


class AgoraAuthorizer(PluralisAuthorizer):
    _node_info: NodeInfo  # Narrow type: guaranteed non-None in this subclass

    def __init__(
        self,
        peer_id: str,
        user_token: str,
        user_email: str,
        role: str,
        auth_server: str,
        node_info: NodeInfo,
        announce_maddrs: str,
        host_port: int,
        check_integrity: bool,
        check_port: bool,
        **kwargs,
    ):
        super().__init__(
            peer_id=peer_id,
            user_token=user_token,
            role=role,
            auth_server=auth_server,
            node_info=node_info,
            **kwargs,
        )

        self._user_email = user_email
        self._check_integrity = check_integrity
        self._check_port = check_port
        self.pipeline_stage = None
        self.monitor_public_key = None
        self.checkpoint_path = None
        self.pushgateway_auth_token = None

        # Parse announce address
        address_parts = announce_maddrs.split("/")
        self._announce_ip_address = address_parts[2]
        self._announce_port = int(address_parts[4])
        self._host_port = host_port

        # Queue polling state
        self._queue_cancel_event = threading.Event()
        self._authorized_event = threading.Event()
        self._queue_thread: threading.Thread | None = None
        self._queue_error: Exception | None = None
        self._queue_info: dict | None = None
        self._state_downloader: StateDownloader | None = None
        self._last_logged_queue_status: tuple[int, int] | None = None

        # Check integrity of files
        if self._check_integrity:
            try:
                integrity_hash = verify_integrity(self._local_private_key)
            except Exception:
                raise RuntimeError("Verification failed.") from None
        else:
            integrity_hash = b"hash"

        # Request parameters
        self.peer_info = {
            "peer_id": self.peer_id,
            "role": self._role,
            "peer_public_key": self.local_public_key.to_bytes().decode(),
            "device": self._node_info.device_name,
            "gpu_memory": self._node_info.gpu_memory,
            "ram": self._node_info.ram,
            "download_speed": self._node_info.download_speed,
            "upload_speed": self._node_info.upload_speed,
            "latency": self._node_info.latency,
            "cloud_provider": self._node_info.cloud_provider,
            "integrity_hash": integrity_hash.decode(),
            "email": self._user_email,
            "announce_ip_address": self._announce_ip_address,
            "announce_port": self._announce_port,
            "hostname": self._hostname,
        }

    def get_authorization(self, response: dict) -> None:
        """Extract authorization information from server response."""
        self._authority_public_key = RSAPublicKey.from_bytes(response["auth_server_public_key"].encode())
        self.monitor_public_key = str(response["monitor_public_key"])
        self.pipeline_stage = response["stage_type"]
        self.expert_uid = response["uid"]
        self._local_access_token = self.extract_token(response)
        self.checkpoint_path = response.get("checkpoint_path", None)
        self.pushgateway_auth_token = response.get("pushgateway_auth_token", None)
        self.exported_role = response.get("exported_role")

        logger.info(
            f"Access for user {self._local_access_token.username} has been granted until {self._local_access_token.expiration_time} UTC"
        )

    def set_state_downloader(self, downloader: StateDownloader) -> None:
        """Set the state downloader to report download progress during queue polling."""
        self._state_downloader = downloader

    def _poll_authorization_status(self, queue_info: dict) -> dict:
        """Poll the authorization status endpoint once.

        Args:
            queue_info: Current queue information containing request_id, session_token.

        Returns:
            Server response (either PeerAccess or PeerQueueInfo).
        """
        url = f"{self._auth_server}/api/authorization_status"

        request_payload: dict = {
            "request_id": queue_info["request_id"],
            "session_token": queue_info["session_token"],
        }

        if self._state_downloader is not None:
            if self._state_downloader.is_finished():
                request_payload["download_finished"] = True
            else:
                eta = self._state_downloader.get_eta()
                if eta >= 0:
                    request_payload["download_eta_seconds"] = eta

        payload = {
            "peer": self.peer_info,
            "request": request_payload,
        }
        response = requests.put(url, headers=self.auth_header, json=payload)
        response.raise_for_status()
        return response.json()

    def _queue_polling_loop(self, initial_queue_info: dict) -> None:
        """Background thread: poll until authorized or error.

        Args:
            initial_queue_info: Initial PeerQueueInfo response from join request.
        """
        queue_info = initial_queue_info

        try:
            while not self._queue_cancel_event.is_set():
                # Check if joins are paused
                if queue_info.get("is_join_paused", False):
                    logger.warning("Node joins are currently paused.")

                sleep_seconds = queue_info.get("next_poll_seconds", 10)

                # Sleep till next poll or cancellation
                if self._queue_cancel_event.wait(timeout=sleep_seconds):
                    raise NonRetriableError("Queue waiting was cancelled.")

                # Poll for status
                try:
                    response = self._poll_authorization_status(queue_info)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code != 503:
                        try:
                            error_detail = e.response.json()["detail"]
                        except Exception:
                            # AWS might return non-JSON error responses, so fallback to a generic message
                            error_detail = "Request is blocked"
                        raise NonRetriableError(error_detail) from None
                    logger.warning(f"Server error during queue poll: {e}. Retrying...")
                    continue
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    logger.warning(f"Network error during queue poll: {e}. Retrying...")
                    continue
                except Exception as e:
                    raise NonRetriableError(str(e)) from None

                # Check if authorized
                if "request_id" not in response:
                    self.get_authorization(response)
                    self._authorized_event.set()
                    return

                # Still in queue - update info
                queue_info = response
                self._queue_info = queue_info
                self._log_queue_status(queue_info)

        except Exception as e:
            self._queue_error = e
            self._authorized_event.set()

    def _log_queue_status(self, queue_info: dict) -> None:
        """Log queue position and estimated wait time."""
        position = queue_info.get("position", 0)
        estimated_wait = queue_info.get("estimated_wait_seconds", 0)

        # Skip logging if nothing changed
        current_status = (position, estimated_wait)
        if current_status == self._last_logged_queue_status:
            return
        self._last_logged_queue_status = current_status

        # Format wait time for human readability
        if estimated_wait >= 3600:
            wait_str = f"{estimated_wait // 3600}h {(estimated_wait % 3600) // 60}m"
        elif estimated_wait >= 60:
            wait_str = f"{estimated_wait // 60}m {estimated_wait % 60}s"
        else:
            wait_str = f"{estimated_wait}s"

        level = (
            logging.DEBUG
            if self._state_downloader is not None and not self._state_downloader.is_finished()
            else logging.INFO
        )
        logger.log(level, f"Authorization queue. Position: {position}, Estimated wait: {wait_str}")

    def wait_for_authorization(self, timeout: float | None = None) -> None:
        """Block until authorization is complete.

        Args:
            timeout: Maximum time to wait in seconds, or None for unlimited.
        """
        authorized = self._authorized_event.wait(timeout=timeout)
        if not authorized:
            raise NonRetriableError(f"Authorization timeout after {timeout}s")
        if self._queue_error is not None:
            raise self._queue_error

    def is_authorized(self) -> bool:
        """Check if authorization is complete (non-blocking)."""
        return self._authorized_event.is_set() and self._queue_error is None

    def get_queue_status(self) -> dict | None:
        """Get current queue information."""
        return self._queue_info

    def cancel_queue_wait(self) -> None:
        """Cancel waiting in the authorization queue."""
        self._queue_cancel_event.set()
        if self._queue_thread is not None and self._queue_thread.is_alive():
            self._queue_thread.join(timeout=5)

    @retry(
        retry=retry_if_not_exception_type(NonRetriableError),
        wait=wait_exponential(multiplier=2),
        stop=stop_after_attempt(7),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def join_experiment(self) -> None:
        """Send authorization request to join the experiment and receive access token or queue information."""
        try:
            url = f"{self._auth_server}/api/join"

            if self._check_port:
                with TestServer(port=self._host_port) as server:
                    response = requests.put(
                        url,
                        headers=self.auth_header,
                        json=self.peer_info,
                    )

                    response.raise_for_status()

                    # Receive server message
                    if not server.wait_for_message(timeout=10):
                        raise NonRetriableError(
                            "Port test failed. Make sure your port forwarding is correct"
                        ) from None
            else:
                response = requests.put(
                    url,
                    headers=self.auth_header,
                    json=self.peer_info,
                )

                response.raise_for_status()

            response = response.json()

            if "request_id" in response:
                # Node is in queue - start background polling
                self._queue_info = response
                self.checkpoint_path = response.get("checkpoint_path", None)
                self._queue_thread = threading.Thread(
                    target=self._queue_polling_loop,
                    args=(response,),
                    daemon=True,
                    name="queue-poller",
                )
                self._queue_thread.start()
            else:
                # Authorized immediately - extract token and info
                self.get_authorization(response)
                self._authorized_event.set()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 503:
                try:
                    error_detail = e.response.json()["detail"]
                except Exception:
                    # AWS might return non-JSON error responses, so fallback to a generic message
                    error_detail = "Request is blocked"
                raise NonRetriableError(error_detail) from None
            raise e
        except requests.exceptions.ConnectionError as e:
            raise e
        except Exception as e:
            raise NonRetriableError(str(e)) from None
