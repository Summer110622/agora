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

import io

import boto3
import requests
import torch


def load_ss_components(
    ss_path: str,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region_name: str | None = None,
) -> dict:
    """Load rcv and fixed_embeddings.

    Args:
        ss_path (str): URL/URI to the subspace compression file.
        aws_access_key_id (str | None, optional): AWS access key ID. Defaults to None.
        aws_secret_access_key (str | None, optional): AWS secret access key. Defaults to None.
        aws_region_name (str | None, optional): AWS region. Defaults to None.

    Raises:
        RuntimeError: Failed to download.

    Returns:
        Dict of: rcv, fixed_tok_weight.
    """
    try:
        is_s3_path = ss_path.startswith("s3://")
        from_s3 = is_s3_path and bool(aws_region_name)
        if is_s3_path and not aws_region_name:
            raise RuntimeError("AWS region must be provided when loading subspace components from S3")

        if from_s3:
            s3_client_kwargs = {"region_name": aws_region_name}
            if aws_access_key_id and aws_secret_access_key:
                s3_client_kwargs["aws_access_key_id"] = aws_access_key_id
                s3_client_kwargs["aws_secret_access_key"] = aws_secret_access_key

            s3_client = boto3.client("s3", **s3_client_kwargs)

            path_without_prefix = ss_path[5:]
            parts = path_without_prefix.split("/", 1)
            response = s3_client.get_object(Bucket=parts[0], Key=parts[1])
            content = response["Body"].read()
        else:
            response = requests.get(ss_path)
            response.raise_for_status()
            content = response.content

        # Load the state dict
        buffer = io.BytesIO(content)
        ss_comp_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    except (requests.RequestException, RuntimeError) as e:
        raise RuntimeError(f"Failed to download subspace compression components: {e}") from e

    return ss_comp_dict
