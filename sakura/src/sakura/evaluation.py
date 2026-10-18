"""Bilingual evaluation accounting and opt-in container-only Python execution.

Container flags follow https://docs.docker.com/engine/containers/resource_constraints/
and https://docs.docker.com/engine/containers/run/. Docker is an explicit runtime
dependency for generated-code execution, never a fallback to host Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping


def bits_per_byte(total_nll_nats: float, utf8_bytes: int) -> float:
    """Total natural-log cross entropy divided by *original* scored UTF-8 bytes.

    Sum token losses first. Do not divide mean token loss by bytes, and do not
    count padding, chat-template headers, or bytes outside the scored corpus.
    """
    if not math.isfinite(total_nll_nats) or total_nll_nats < 0:
        raise ValueError("total_nll_nats must be finite and nonnegative")
    if not isinstance(utf8_bytes, int) or isinstance(utf8_bytes, bool) or utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be a positive integer")
    return total_nll_nats / (math.log(2) * utf8_bytes)


def _validated_results(results: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key = {}
    for raw in results:
        value = dict(raw)
        language, problem = value.get("language"), value.get("problem_id")
        if language not in {"ja", "en"} or not isinstance(problem, str) or not problem:
            raise ValueError("evaluation results require language ja/en and a nonempty problem_id")
        key = (problem, language)
        if key in by_key:
            raise ValueError("pass@1 requires exactly one sample per problem/language")
        executed = value.get("executed", False)
        passed = value.get("passed")
        if not isinstance(executed, bool) or (executed and not isinstance(passed, bool)):
            raise ValueError("executed results require a boolean passed value")
        if not executed and passed is not None:
            raise ValueError("an unexecuted sample cannot claim passed or failed")
        nll, byte_count = value.get("nll_nats"), value.get("utf8_bytes")
        if (nll is None) != (byte_count is None):
            raise ValueError("nll_nats and utf8_bytes must be provided together")
        if nll is not None:
            bits_per_byte(nll, byte_count)
        value["executed"] = executed
        value["passed"] = passed
        by_key[key] = value
    return by_key


def summarize_evaluation(
    results: Iterable[Mapping[str, Any]], baseline: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate one completion per ja/en task and matched base-model regressions.

    Unexecuted samples remain excluded, never counted as successes or failures.
    Paired metrics use only problems with executed Japanese *and* English tasks.
    Baseline comparisons use the intersection of actually executed sample keys.
    """
    by_key = _validated_results(results)
    paired = sorted(problem for problem, language in by_key
                    if language == "ja" and by_key[(problem, "ja")]["executed"]
                    and (problem, "en") in by_key and by_key[(problem, "en")]["executed"])
    per_language = {}
    for language in ("ja", "en"):
        samples = [r for (_, lang), r in by_key.items() if lang == language]
        executed = [r for r in samples if r["executed"]]
        scored = [r for r in samples if r.get("nll_nats") is not None]
        nll, byte_count = sum(r["nll_nats"] for r in scored), sum(r["utf8_bytes"] for r in scored)
        passed = sum(r["passed"] for r in executed)
        per_language[language] = {"samples": len(samples), "executed": len(executed), "passed": passed,
                                  "pass_at_1": passed / len(executed) if executed else None,
                                  "total_nll_nats": nll, "utf8_bytes": byte_count,
                                  "bits_per_byte": bits_per_byte(nll, byte_count) if byte_count else None}
    paired_rates = {lang: sum(by_key[(problem, lang)]["passed"] for problem in paired) / len(paired) if paired else None
                    for lang in ("ja", "en")}
    total_bytes = sum(value["utf8_bytes"] for value in per_language.values())
    total_nll = sum(value["total_nll_nats"] for value in per_language.values())
    report = {"per_language": per_language,
              "paired": {"problems": len(paired), "problem_ids": paired, "pass_at_1": paired_rates,
                         "balanced_pass_at_1": (paired_rates["ja"] + paired_rates["en"]) / 2 if paired else None,
                         "both_passed": sum(all(by_key[(problem, lang)]["passed"] for lang in ("ja", "en")) for problem in paired)},
              "total_nll_nats": total_nll, "utf8_bytes": total_bytes,
              "bits_per_byte": bits_per_byte(total_nll, total_bytes) if total_bytes else None}
    if baseline is not None:
        original = _validated_results(baseline)
        matched = sorted(key for key in by_key if key in original and by_key[key]["executed"] and original[key]["executed"])
        regressions = [key for key in matched if original[key]["passed"] and not by_key[key]["passed"]]
        improvements = [key for key in matched if not original[key]["passed"] and by_key[key]["passed"]]
        report["comparison"] = {
            "matched_executed_samples": len(matched),
            "baseline_pass_at_1": sum(original[k]["passed"] for k in matched) / len(matched) if matched else None,
            "finetuned_pass_at_1": sum(by_key[k]["passed"] for k in matched) / len(matched) if matched else None,
            "regressions": [{"problem_id": p, "language": lang} for p, lang in regressions],
            "improvements": [{"problem_id": p, "language": lang} for p, lang in improvements],
            "unmatched_or_unexecuted": len(set(by_key) | set(original)) - len(matched),
        }
    return report


@dataclass(frozen=True)
class SandboxLimits:
    # Require an already-installed immutable image; no implicit pulls or mutable
    # tags. A project administrator selects and audits this image before opting in.
    image: str | None = None
    timeout_seconds: float = 10
    memory_mb: int = 256
    cpus: float = 1
    pids: int = 64
    output_bytes: int = 65536
    source_bytes: int = 262144

    def validate(self) -> None:
        if not isinstance(self.image, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[0-9a-f]{64}", self.image):
            raise ValueError("sandbox execution requires an explicit locally installed image@sha256:<64 hex> digest")
        for name in ("timeout_seconds", "memory_mb", "cpus", "pids", "output_bytes", "source_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"sandbox {name} must be positive and finite")
        if any(not isinstance(getattr(self, name), int) for name in ("memory_mb", "pids", "output_bytes", "source_bytes")):
            raise ValueError("sandbox memory, pids and byte limits must be integers")


def _bounded_process(command: list[str], payload: bytes, limits: SandboxLimits) -> tuple[int | None, str, str, str]:
    """Drain container output with bounded memory; the caller always removes it."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers: list[bytearray] = [bytearray(), bytearray()]
    exceeded = threading.Event()

    def drain(stream: Any, target: bytearray) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                remaining = limits.output_bytes - len(target)
                target.extend(chunk[:max(remaining, 0)])
                if len(chunk) > remaining:
                    exceeded.set()
        finally:
            stream.close()

    readers = [threading.Thread(target=drain, args=(process.stdout, buffers[0]), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, buffers[1]), daemon=True)]
    for reader in readers:
        reader.start()

    def write_input() -> None:
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    writer = threading.Thread(target=write_input, daemon=True)
    writer.start()
    deadline = time.monotonic() + limits.timeout_seconds
    status = "completed"
    try:
        while process.poll() is None:
            if exceeded.is_set():
                status = "output_limit"
                process.kill()
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                process.kill()
                break
            time.sleep(0.02)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
        for reader in readers:
            reader.join(timeout=1)
        writer.join(timeout=1)
    if exceeded.is_set() and status == "completed":
        status = "output_limit"
    return process.returncode, buffers[0].decode("utf-8", errors="replace"), buffers[1].decode("utf-8", errors="replace"), status


def evaluate_python(
    code: str, test_code: str, *, execute: bool = False, limits: SandboxLimits | None = None,
) -> dict[str, Any]:
    """Run generated Python only inside an explicitly configured Docker sandbox.

    Tests must be trusted benchmark assertions. This provides OS/resource isolation;
    it is not a proof that hostile code cannot manipulate Python-level test objects.
    No host interpreter, mounts, host network, privileged mode or image pulls exist.
    """
    if not execute:
        return {"status": "skipped", "executed": False, "passed": None,
                "reason": "generated-code execution is opt-in and requires a digest-pinned local Docker image"}
    limits = limits or SandboxLimits()
    limits.validate()
    if not isinstance(code, str) or not isinstance(test_code, str) or not test_code.strip():
        raise ValueError("code must be text and test_code must contain benchmark assertions")
    if len(code.encode("utf-8")) + len(test_code.encode("utf-8")) > limits.source_bytes:
        raise ValueError("code and tests exceed the configured source byte limit")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for generated-code execution; host execution is disabled")
    try:
        installed = subprocess.run([docker, "image", "inspect", limits.image], capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("cannot inspect local Docker sandbox image") from error
    if installed.returncode:
        raise RuntimeError("sandbox image digest is not locally installed or Docker is unavailable; automatic pulling is disabled")
    name = "sakura-eval-" + secrets.token_hex(12)
    sentinel = "SAKURA_TESTS_COMPLETED_" + secrets.token_hex(24)
    # Compile each unit independently so a future import in the completion cannot
    # break the harness and an early sys.exit(0) cannot be mistaken for success.
    script = ("namespace = {'__name__': '__main__'}\n"
              f"exec(compile({json.dumps(code, ensure_ascii=True)}, '<completion>', 'exec'), namespace)\n"
              f"exec(compile({json.dumps(test_code, ensure_ascii=True)}, '<tests>', 'exec'), namespace)\n"
              f"print({sentinel!r})\n").encode("utf-8")
    command = [docker, "run", "--rm", "--pull=never", "--name", name, "--network=none", "--read-only",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
               f"--memory={limits.memory_mb}m", f"--memory-swap={limits.memory_mb}m", f"--cpus={limits.cpus}",
               f"--pids-limit={limits.pids}", "--ulimit=nofile=64:64", "--ulimit=fsize=1048576:1048576",
               "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777", "--workdir=/tmp", "--log-driver=none",
               "--env=PYTHONDONTWRITEBYTECODE=1", "--entrypoint=python", "-i", limits.image, "-I", "-B", "-"]
    cleanup_ok = True
    try:
        returncode, stdout, stderr, status = _bounded_process(command, script, limits)
    finally:
        try:
            cleanup = subprocess.run([docker, "rm", "--force", name], capture_output=True, timeout=15, check=False)
            # --rm normally removed it already; a missing container is success.
            cleanup_ok = cleanup.returncode == 0 or b"No such container" in cleanup.stderr
        except (OSError, subprocess.TimeoutExpired):
            cleanup_ok = False
    if not cleanup_ok:
        raise RuntimeError(f"sandbox container cleanup failed; inspect and remove {name}")
    if returncode == 125:
        raise RuntimeError("Docker could not start the sandbox; generated code was not evaluated")
    passed = status == "completed" and returncode == 0 and sentinel in stdout.splitlines()
    return {"status": status, "executed": True, "passed": passed, "returncode": returncode,
            "stdout": stdout.replace(sentinel + "\n", ""), "stderr": stderr, "image": limits.image}
