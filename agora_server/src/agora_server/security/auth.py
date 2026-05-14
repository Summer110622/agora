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

import os
import sys

from typing import TypeVar

from agora_server.security.authorizer import PluralisAuthorizer
from agora_server.utils.node_info import NodeInfo
from cryptography.hazmat.primitives import serialization

from hivemind import PeerID
from hivemind.proto import crypto_pb2
from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.logging import get_logger


AuthorizerType = TypeVar("AuthorizerType", bound=PluralisAuthorizer)

logger = get_logger(__name__)


def save_identity(private_key: RSAPrivateKey, identity_path: str) -> None:
    """Save private key to file.

    Args:
        private_key (RSAPrivateKey): Local private key.
        identity_path (str): Path to save the key.

    Raises:
        FileNotFoundError: Can't create file.
    """
    protobuf = crypto_pb2.PrivateKey(key_type=crypto_pb2.KeyType.RSA, data=private_key.to_bytes())

    try:
        with open(identity_path, "wb") as f:
            f.write(protobuf.SerializeToString())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The directory `{os.path.dirname(identity_path)}` for saving the identity does not exist"
        ) from None
    os.chmod(identity_path, 0o400)


def authorize_with_pluralis(
    user_token: str,
    role: str,
    auth_server: str,
    identity_path: str,
    node_info: NodeInfo | None = None,
    authorizer_cls: type[AuthorizerType] = PluralisAuthorizer,
    expert_uid: str | None = None,
    **kwargs,
) -> AuthorizerType:
    """Generate local keys and send authorization request to join the run.

    Args:
        user_token (str): Authentication token.
        role (str): Role in the swarm.
        auth_server (str): Authorization server URL.
        identity_path (str): Path to save/load private key.
        node_info (NodeInfo | None): Information about the node. Defaults to None.
        authorizer_cls (type[AuthorizerType]): Class to instantiate the authorizer. Defaults to PluralisAuthorizer.
        expert_uid (str | None): Expert UID (e.g. head.0.42). Defaults to None.

    Returns:
        AuthorizerType: Authorizer instance.
    """
    logger.info("Authorization started...")

    # Generate private key or read from file
    local_private_key = RSAPrivateKey.process_wide()

    if os.path.exists(identity_path):
        with open(identity_path, "rb") as f:
            key_data = crypto_pb2.PrivateKey.FromString(f.read()).data
            private_key = serialization.load_der_private_key(key_data, password=None)
            if local_private_key._process_wide_key:
                local_private_key._process_wide_key._private_key = private_key
            else:
                logger.error("Failed to initialize process-wide private key")
                raise RuntimeError("Process-wide key is None")
    else:
        save_identity(local_private_key, identity_path)

    # Get static peer id
    with open(identity_path, "rb") as f:
        peer_id = str(PeerID.from_identity(f.read()))

    # Authorize
    try:
        authorizer = authorizer_cls(
            peer_id=peer_id,
            user_token=user_token,
            role=role,
            auth_server=auth_server,
            node_info=node_info,
            expert_uid=expert_uid,
            **kwargs,
        )

        authorizer.join_experiment()
        return authorizer
    except Exception as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        sys.exit(1)
