"""Direct-address discovery, plus an optional authenticated Hivemind DHT adapter."""
from __future__ import annotations

import base64
import math
import time
import urllib.parse
from typing import Mapping

from cryptography.exceptions import InvalidSignature

from .auth import Identity, MessageAuthenticator, canonical_json


def _endpoint(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError("Worker endpoint must be an explicit HTTP(S) URL")
    return url


class StaticDiscovery:
    """Works on Windows and needs no Hivemind/native daemon installation."""
    def __init__(self, endpoints: Mapping[str, str], *, allowed_members: Mapping[str, str] | None = None):
        if allowed_members is not None and set(endpoints) - set(allowed_members):
            raise ValueError("Endpoint configuration contains an unknown member")
        self.endpoints = {member: _endpoint(url) for member, url in endpoints.items()}

    def resolve(self) -> dict[str, str]:
        return dict(self.endpoints)

    def close(self) -> None:
        pass


class HivemindDiscovery:
    """Lazy optional discovery using the repository's pinned Hivemind installation.

    Discovery is only an address hint. Every RPC remains independently authenticated.
    Re-reading a signed DHT record is permitted until its expiry; it is not an RPC.
    """
    def __init__(self, initial_peers: list[str], identity: Identity,
                 allowed_members: Mapping[str, str], run_hash: str, *, host_maddrs: list[str] | None = None):
        try:
            import hivemind
        except ImportError as error:
            raise RuntimeError("Hivemind discovery is optional; use StaticDiscovery on this installation") from error
        self.auth = MessageAuthenticator(identity, allowed_members, run_hash)
        options = {"initial_peers": list(initial_peers), "start": True}
        if host_maddrs is not None:
            options["host_maddrs"] = host_maddrs
        self.dht = hivemind.DHT(**options)
        self.key = f"sakura.{run_hash}.http-workers.v1"

    def publish(self, url: str, *, ttl: float = 90) -> None:
        if not 1 <= ttl <= 3600:
            raise ValueError("Discovery TTL must be between 1 and 3600 seconds")
        now = time.time()
        record = {"domain": "sakura-discovery-v1", "member_id": self.auth.identity.member_id,
                  "run_hash": self.auth.run_hash, "url": _endpoint(url),
                  "published": now, "expires": now + ttl}
        signature = self.auth.identity.private_key.sign(canonical_json(record))
        record["signature"] = base64.b64encode(signature).decode("ascii")
        self.dht.store(self.key, subkey=self.auth.identity.member_id, value=record,
                       expiration_time=now + ttl)

    def resolve(self) -> dict[str, str]:
        found = self.dht.get(self.key, latest=True)
        values = getattr(found, "value", {}) if found is not None else {}
        if not isinstance(values, dict):
            return {}
        endpoints: dict[str, str] = {}
        for member, wrapped in values.items():
            record = getattr(wrapped, "value", wrapped)
            try:
                if not isinstance(record, dict) or set(record) != {
                    "domain", "member_id", "run_hash", "url", "published", "expires", "signature"
                }:
                    continue
                if (record["domain"] != "sakura-discovery-v1" or record["member_id"] != member
                        or member not in self.auth.members or record["run_hash"] != self.auth.run_hash):
                    continue
                published, expires = record["published"], record["expires"]
                if (type(published) not in (float, int) or type(expires) not in (float, int)
                        or not math.isfinite(published) or not math.isfinite(expires)
                        or published > time.time() + 120 or expires < time.time()
                        or not 0 < expires - published <= 3600):
                    continue
                unsigned = {key: value for key, value in record.items() if key != "signature"}
                self.auth.members[member].verify(base64.b64decode(record["signature"], validate=True),
                                                 canonical_json(unsigned))
                endpoints[member] = _endpoint(record["url"])
            except (ValueError, TypeError, InvalidSignature):
                continue
        return endpoints

    def close(self) -> None:
        self.dht.shutdown()
