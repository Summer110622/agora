"""Bounded reverse-poll inference queue. Colab needs only outbound HTTPS.

The existing signed RPC authenticates both peers; ``pass_member=True`` passes
the verified signer into handle(), so request fields cannot impersonate roles.
No jobs, prompts, or keys are persisted. A restart loses pending requests.
"""
from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from urllib.parse import urlsplit

from .auth import canonical_json


def validate_endpoint(url: str, *, allow_insecure_http: bool = False) -> str:
    value = urlsplit(url)
    if value.scheme not in {"http", "https"} or not value.hostname or value.username or value.password:
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials")
    if value.path != "/rpc" or value.query or value.fragment:
        raise ValueError("Endpoint must end in /rpc without query or fragment")
    if value.scheme != "https" and value.hostname not in {"127.0.0.1", "::1", "localhost"} and not allow_insecure_http:
        raise ValueError("Remote inference requires HTTPS; trusted private HTTP needs --allow-insecure-http")
    return url


def validate_generation(prompt, max_new_tokens, *, max_tokens: int = 2048) -> None:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 65536:
        raise ValueError("Prompt must contain 1..65536 UTF-8 bytes")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= max_tokens:
        raise ValueError(f"max_new_tokens must be in 1..{max_tokens}")


class InferenceGateway:
    def __init__(self, pc_member: str, worker_member: str, *, max_jobs: int = 16,
                 job_timeout: float = 600, result_ttl: float = 300, clock=time.monotonic):
        if pc_member == worker_member:
            raise ValueError("PC and GPU worker must have different identities")
        if type(max_jobs) is not int or not 1 <= max_jobs <= 128:
            raise ValueError("max_jobs must be in 1..128")
        if not all(math.isfinite(v) and v > 0 for v in (job_timeout, result_ttl)):
            raise ValueError("Timeouts must be finite and positive")
        self.pc_member, self.worker_member = pc_member, worker_member
        self.max_jobs, self.job_timeout, self.result_ttl = max_jobs, job_timeout, result_ttl
        self.clock, self.jobs = clock, {}
        self.condition = threading.Condition()

    def _expire(self):
        now = self.clock()
        for identifier, row in list(self.jobs.items()):
            if row["status"] in {"queued", "running"} and now >= row["deadline"]:
                row.update(status="failed", error="Generation deadline expired; submit a new job", finished=now)
            if row["status"] in {"done", "failed"} and now - row["finished"] >= self.result_ttl:
                del self.jobs[identifier]

    def handle(self, request: dict, member: str) -> dict:
        op = request.get("op")
        expected = self.worker_member if op in {"take", "complete"} else self.pc_member
        if member != expected:
            raise PermissionError("Authenticated member cannot perform this inference operation")
        with self.condition:
            self._expire()
            if op == "info":
                return {"service": "sakura-reverse-inference", "jobs": len(self.jobs),
                        "max_jobs": self.max_jobs, "worker_member": self.worker_member}
            if op == "submit":
                prompt, count = request.get("prompt"), request.get("max_new_tokens")
                validate_generation(prompt, count)
                identifier = request.get("job_id")
                if not isinstance(identifier, str) or not 16 <= len(identifier) <= 128 or not identifier.isalnum():
                    raise ValueError("job_id must be 16..128 alphanumeric characters")
                digest = hashlib.sha256(canonical_json({"prompt": prompt, "max_new_tokens": count})).hexdigest()
                if identifier in self.jobs:
                    if self.jobs[identifier]["digest"] != digest:
                        raise ValueError("job_id already belongs to another prompt")
                    return {"job_id": identifier, "status": self.jobs[identifier]["status"]}
                if len(self.jobs) >= self.max_jobs:
                    raise RuntimeError("Inference queue is full; wait for results to expire")
                self.jobs[identifier] = {"status": "queued", "prompt": prompt, "max_new_tokens": count,
                                         "digest": digest, "deadline": self.clock() + self.job_timeout}
                self.condition.notify_all()
                return {"job_id": identifier, "status": "queued"}
            if op == "take":
                wait = request.get("wait_seconds", 10)
                if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait) or not 0 <= wait <= 20:
                    raise ValueError("Polling wait must be in 0..20 seconds")
                end = self.clock() + wait
                while True:
                    self._expire()
                    # Serialize GPU work, including across accidentally duplicated workers.
                    if not any(row["status"] == "running" for row in self.jobs.values()):
                        for identifier, row in self.jobs.items():
                            if row["status"] == "queued":
                                row.update(status="running", lease=secrets.token_hex(32))
                                return {"job": {"job_id": identifier, "lease": row["lease"],
                                                "prompt": row["prompt"], "max_new_tokens": row["max_new_tokens"]}}
                    remaining = end - self.clock()
                    if remaining <= 0:
                        return {"job": None}
                    self.condition.wait(timeout=remaining)
            if op in {"get", "complete"}:
                identifier = request.get("job_id")
                if not isinstance(identifier, str) or identifier not in self.jobs:
                    raise KeyError("Unknown or expired inference job")
                row = self.jobs[identifier]
                if op == "get":
                    return {key: row[key] for key in ("status", "result", "error") if key in row}
                lease = request.get("lease")
                if not isinstance(lease, str) or not secrets.compare_digest(row.get("lease", ""), lease):
                    raise PermissionError("Wrong generation lease")
                result, error = request.get("result"), request.get("error")
                if result is not None and (not isinstance(result, dict) or not isinstance(result.get("text"), str)):
                    raise ValueError("Generation result must include text")
                if (result is None) == (error is None):
                    raise ValueError("Complete with exactly one result or error")
                if error is not None and (not isinstance(error, str) or len(error) > 2048):
                    raise ValueError("Invalid worker error")
                completed = {"result": result} if result is not None else {"error": error}
                digest = hashlib.sha256(canonical_json(completed)).hexdigest()
                if len(canonical_json(completed)) > 1024 * 1024:
                    raise ValueError("Inference result exceeds 1MiB")
                if row["status"] in {"done", "failed"}:
                    if row.get("completion_digest") == digest:
                        return {"accepted": True}
                    raise ValueError("Generation already finished or expired")
                if row["status"] != "running":
                    raise ValueError("Generation has not been leased")
                row.update(status="done" if result is not None else "failed", finished=self.clock(),
                           completion_digest=digest, **completed)
                self.condition.notify_all()
                return {"accepted": True}
            raise ValueError("Unknown inference gateway operation")


def request_generation(call, prompt: str, max_new_tokens: int = 64, *, timeout: float = 600,
                       poll_interval: float = .25) -> dict:
    validate_generation(prompt, max_new_tokens)
    if not math.isfinite(timeout) or timeout <= 0 or not 0 < poll_interval <= 10:
        raise ValueError("Invalid client timeout or polling interval")
    start = time.perf_counter()
    identifier = secrets.token_hex(16)
    call({"op": "submit", "job_id": identifier, "prompt": prompt, "max_new_tokens": max_new_tokens})
    while True:
        reply = call({"op": "get", "job_id": identifier})
        if reply["status"] == "done":
            result = dict(reply["result"])
            result["end_to_end_seconds"] = time.perf_counter() - start
            return result
        if reply["status"] == "failed":
            raise RuntimeError(reply.get("error", "GPU worker failed"))
        if time.perf_counter() - start >= timeout:
            raise TimeoutError("Inference request timed out; no automatic resubmission")
        time.sleep(poll_interval)


def process_one_job(call, generator, *, wait_seconds: float = 10) -> bool:
    """Run at most one generation. Lost responses never trigger a second generation."""
    job = call({"op": "take", "wait_seconds": wait_seconds})["job"]
    if job is None:
        return False
    complete = {"op": "complete", "job_id": job["job_id"], "lease": job["lease"]}
    try:
        complete["result"] = generator.generate(job["prompt"], max_new_tokens=job["max_new_tokens"])
    except Exception as error:
        complete["error"] = f"{type(error).__name__}: {str(error)[:1900]}"
    # Completion is idempotent for this lease; retry only the acknowledgement.
    from .transport import RpcError
    for attempt in range(3):
        try:
            call(complete)
            return True
        except RpcError:
            if attempt == 2:
                raise
            time.sleep(1)
