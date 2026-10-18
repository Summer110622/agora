"""Launch two real worker processes, train, restart, resume and export on loopback.

Run with the pinned environment from the repository root. This is a local
multi-process integration exercise, not evidence of multiple physical hosts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sakura" / "src"))

from sakura.auth import Identity
from sakura.manifest import load_manifest, run_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    if directory.exists():
        raise FileExistsError("Use a new smoke output directory")
    directory.mkdir(parents=True)
    public_files = []
    for member in ("trainer", "worker-0", "worker-1"):
        identity = Identity.generate(member)
        identity.save(directory / f"{member}.private.json")
        public = directory / f"{member}.public.json"
        public.write_text(json.dumps({"member_id": member, "public_key": identity.public_key}), encoding="utf-8")
        public_files.append(str(public))
    launcher = [sys.executable, "-u", str(ROOT / "sakura_cli.py")]
    def run(arguments, name):
        result = subprocess.run([*launcher, *arguments], cwd=ROOT, text=True, encoding="utf-8",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (directory / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (directory / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name}: {result.stderr[-2000:]}")
        return result
    run(["init", "--directory", str(directory / "run"), "--members", *public_files], "init")
    manifest_path = directory / "run" / "run.json"
    config = load_manifest(manifest_path)
    processes, streams, endpoints = [], [], []
    def start_workers():
        endpoints.clear()
        for index in range(2):
            output = (directory / f"worker-{index}.stdout.log").open("w", encoding="utf-8")
            error = (directory / f"worker-{index}.stderr.log").open("w", encoding="utf-8")
            streams.extend([output, error])
            process = subprocess.Popen([*launcher, "join", "--manifest", str(manifest_path), "--identity",
                                        str(directory / f"worker-{index}.private.json"), "--stage", str(index),
                                        "--port", "0"], cwd=ROOT, stdout=output, stderr=error,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append(process)
        deadline = time.monotonic() + 90
        while len(endpoints) != 2:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A worker exited; inspect worker stderr logs")
            endpoints.clear()
            for index in range(2):
                try:
                    info = json.loads((directory / f"worker-{index}.stdout.log").read_text(encoding="utf-8"))
                    endpoints.append({"stage_index": index, "member_id": info["member_id"], "url": info["url"]})
                except (ValueError, KeyError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Workers did not start within 90 seconds")
            time.sleep(0.2)
        (directory / "endpoints.json").write_text(json.dumps(endpoints), encoding="utf-8")
    def stop_workers():
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        processes.clear()
        for stream in streams:
            stream.close()
        streams.clear()
    try:
        start_workers()
        common = ["--manifest", str(manifest_path), "--endpoints", str(directory / "endpoints.json"),
                  "--identity", str(directory / "trainer.private.json")]
        run(["train", *common, "--steps", "3"], "train")
        stop_workers()
        start_workers()
        run(["resume", *common, "--steps", "2"], "resume")
        run(["export", "--manifest", str(manifest_path), "--output", str(directory / "adapter")], "export")
        summary = {"scope": "two separate CPU worker processes on one physical host", "signed_http": True,
                   "run_hash": run_hash(config), "steps_before_restart": 3, "steps_after_resume": 5,
                   "adapter_exported": True, "fixture_only": True}
        (directory / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stop_workers()


if __name__ == "__main__":
    main()
