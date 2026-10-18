"""Colab/PC inference commands. Independent from the training launcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import threading

from .auth import Identity, manifest_hash
from .inference_gateway import (InferenceGateway, process_one_job, request_generation, validate_endpoint)


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def load_inference_manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("purpose") != "sakura-inference":
        raise ValueError("Expected a Sakura inference manifest; training run.json is only for split commands")
    if not isinstance(value.get("model_id"), str) or not value["model_id"].strip():
        raise ValueError("model_id is required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("revision"))):
        raise ValueError("Inference model/tokenizer revision must be an immutable commit SHA")
    if type(value.get("max_context")) is not int or not 2 <= value["max_context"] <= 131072:
        raise ValueError("Invalid inference context limit")
    members = value.get("members", {})
    roles = value.get("roles", {})
    if not isinstance(members, dict) or not isinstance(roles, dict) or set(roles) != {"pc", "worker"}:
        raise ValueError("Inference manifest requires PC and worker roles")
    if len(members) != 2 or roles["pc"] == roles["worker"] or set(roles.values()) != set(members):
        raise ValueError("Inference roles must reference two distinct registered identities")
    manifest_hash(value)
    return value


def _init(args):
    from .inference_generate import DEFAULT_MODEL, DEFAULT_REVISION
    if args.model != DEFAULT_MODEL and args.revision == DEFAULT_REVISION:
        raise ValueError("A different model requires its own --revision COMMIT_SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision requires an immutable 40-character commit SHA")
    if type(args.max_context) is not int or not 2 <= args.max_context <= 131072:
        raise ValueError("--max-context must be in 2..131072")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    identities = [Identity.generate("pc"), Identity.generate("colab")]
    value = {"schema_version": 1, "purpose": "sakura-inference", "model_id": args.model,
             "revision": args.revision, "max_context": args.max_context,
             "roles": {"pc": "pc", "worker": "colab"},
             "members": {identity.member_id: identity.public_key for identity in identities}}
    for identity in identities:
        identity.save(directory / f"{identity.member_id}.private.json")
    manifest = directory / "inference-run.json"
    manifest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_inference_manifest(manifest)
    _print({"manifest": str(manifest.resolve()), "run_hash": manifest_hash(value),
            "upload_to_colab": ["inference-run.json", "colab.private.json"],
            "keep_on_pc": "pc.private.json"})


def _identity(args, manifest, role):
    identity = Identity.load(args.identity)
    if identity.member_id != manifest["roles"][role] or identity.public_key != manifest["members"][identity.member_id]:
        raise ValueError(f"This command requires the registered {role} private identity")
    return identity


def _call(args, manifest, identity):
    from .transport import RpcClient
    endpoint = validate_endpoint(args.endpoint, allow_insecure_http=args.allow_insecure_http)
    client = RpcClient(identity, manifest["members"], manifest_hash(manifest), timeout=30)
    return lambda request: client.call(endpoint, request, expected_member=manifest["roles"]["pc"])


def _wait_for_shutdown(server):
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


def _gateway(args):
    from .transport import RpcServer
    manifest = load_inference_manifest(args.manifest)
    identity = _identity(args, manifest, "pc")
    gateway = InferenceGateway(manifest["roles"]["pc"], manifest["roles"]["worker"],
                               max_jobs=args.max_jobs, job_timeout=args.job_timeout)
    server = RpcServer(gateway.handle, identity, manifest["members"], manifest_hash(manifest),
                       host=args.host, port=args.port, allow_external=args.allow_external, pass_member=True,
                       tls_certfile=args.tls_cert, tls_keyfile=args.tls_key,
                       max_blob_bytes=64 * 1024 * 1024).start()
    _print({"local_endpoint": server.url, "run_hash": manifest_hash(manifest),
            "connection": "Colab polls your reachable HTTPS /rpc endpoint; no inbound Colab port"})
    _wait_for_shutdown(server)


def _worker(args):
    from .inference_generate import HFGenerator
    manifest = load_inference_manifest(args.manifest)
    identity = _identity(args, manifest, "worker")
    call = _call(args, manifest, identity)
    generator = HFGenerator.from_pretrained(manifest["model_id"], manifest["revision"], device=args.device,
                                            dtype=args.dtype, max_context=manifest["max_context"],
                                            chat_template=not args.raw_prompt)
    _print({"status": "ready", "device": str(generator.device), "run_hash": manifest_hash(manifest)})
    try:
        while True:
            process_one_job(call, generator)
    except KeyboardInterrupt:
        return


def _chat(args):
    manifest = load_inference_manifest(args.manifest)
    identity = _identity(args, manifest, "pc")
    _print(request_generation(_call(args, manifest, identity), args.prompt, args.max_new_tokens, timeout=args.timeout))


def _local(args):
    from .inference_generate import DEFAULT_MODEL, DEFAULT_REVISION, HFGenerator
    if args.model != DEFAULT_MODEL and args.revision == DEFAULT_REVISION:
        raise ValueError("A different model requires its own --revision COMMIT_SHA")
    generator = HFGenerator.from_pretrained(args.model, args.revision, device=args.device, dtype=args.dtype,
                                            max_context=args.max_context, chat_template=not args.raw_prompt)
    result = generator.generate(args.prompt, args.max_new_tokens)
    if args.benchmark:
        result["benchmark"] = generator.benchmark(args.prompt, args.max_new_tokens, repeats=args.repeats)
    _print(result)


def _split_serve(args):
    from .manifest import load_manifest, run_hash
    from .inference import CachedStageWorker
    from .transport import RpcServer
    manifest = load_manifest(args.manifest)
    identity = Identity.load(args.identity)
    worker = CachedStageWorker(manifest, args.stage, args.device, args.dtype, base_path=args.base_path,
                               adapter_path=args.adapter_path, max_context=args.max_context,
                               max_cache_bytes=args.max_cache_mib * 1024 * 1024)
    server = RpcServer(worker.handle, identity, manifest["members"], run_hash(manifest),
                       host=args.host, port=args.port, allow_external=args.allow_external,
                       tls_certfile=args.tls_cert, tls_keyfile=args.tls_key).start()
    _print({"endpoint": server.url, "stage": args.stage, "run_hash": run_hash(manifest)})
    _wait_for_shutdown(server)


def _split_chat(args):
    import torch
    from .manifest import load_manifest, run_hash
    from .inference import CachedPipeline
    from .tokenization import load_tokenizer
    from .trainer import StageEndpoint
    from .transport import RpcClient
    manifest = load_manifest(args.manifest)
    tokenizer = load_tokenizer(manifest["tokenizer"], base_path=args.base_path)
    configured = json.loads(Path(args.endpoints).read_text(encoding="utf-8"))
    if not isinstance(configured, list) or sorted(row["stage_index"] for row in configured) != list(range(manifest["num_stages"])):
        raise ValueError("endpoints.json must contain each stage exactly once")
    client = RpcClient(Identity.load(args.identity), manifest["members"], run_hash(manifest), timeout=args.timeout)
    endpoints = []
    for row in configured:
        validate_endpoint(row["url"], allow_insecure_http=args.allow_insecure_http)
        endpoints.append(StageEndpoint(row["stage_index"],
                        lambda request, row=row: client.call(row["url"], request, expected_member=row["member_id"])))
    if not args.raw_prompt and getattr(tokenizer, "chat_template", None):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                               tokenize=True, add_generation_prompt=True)
    else:
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    result = CachedPipeline(endpoints, max_context=args.max_context, expected_run_hash=run_hash(manifest)).generate(
        torch.tensor([tokens], dtype=torch.long), args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    _print({"text": tokenizer.decode(result["tokens"][0].tolist(), skip_special_tokens=True), "metrics": result["metrics"]})


def parser():
    from .inference_generate import DEFAULT_MODEL, DEFAULT_REVISION
    root = argparse.ArgumentParser(description="Cached inference between a PC and eligible Colab/owned GPU runtimes")
    sub = root.add_subparsers(dest="command", required=True)

    def model_options(command):
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--revision", default=DEFAULT_REVISION)
        command.add_argument("--max-context", type=int, default=4096)

    def hardware(command):
        command.add_argument("--device", default="cpu")
        command.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
        command.add_argument("--raw-prompt", action="store_true")

    def identity(command):
        command.add_argument("--manifest", required=True)
        command.add_argument("--identity", required=True)

    def network(command):
        command.add_argument("--endpoint", required=True)
        command.add_argument("--allow-insecure-http", action="store_true")

    def serving(command):
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=49201)
        command.add_argument("--allow-external", action="store_true")
        command.add_argument("--tls-cert", help="Publicly trusted PEM certificate chain for direct HTTPS")
        command.add_argument("--tls-key", help="PEM private key; keep outside version control")

    init = sub.add_parser("init", help="Create public inference manifest and two private identities on PC")
    init.add_argument("--directory", required=True); model_options(init); init.set_defaults(function=_init)
    gateway = sub.add_parser("gateway", help="PC reverse-poll queue; requires your reachable HTTPS endpoint")
    identity(gateway); serving(gateway)
    gateway.add_argument("--max-jobs", type=int, default=16)
    gateway.add_argument("--job-timeout", type=float, default=600); gateway.set_defaults(function=_gateway)
    worker = sub.add_parser("worker", help="Colab GPU worker; only outbound connection to the PC gateway")
    identity(worker); network(worker); hardware(worker); worker.set_defaults(function=_worker)
    chat = sub.add_parser("chat", help="Send a prompt from PC and report GPU and end-to-end timings")
    identity(chat); network(chat); chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-new-tokens", type=int, default=64)
    chat.add_argument("--timeout", type=float, default=600); chat.set_defaults(function=_chat)
    local = sub.add_parser("local", help="Single-runtime cached generation; also usable in free Colab notebooks")
    model_options(local); hardware(local); local.add_argument("--prompt", required=True)
    local.add_argument("--max-new-tokens", type=int, default=64)
    local.add_argument("--benchmark", action="store_true"); local.add_argument("--repeats", type=int, default=3)
    local.set_defaults(function=_local)
    serve = sub.add_parser("split-serve", help="Serve one cached Qwen3.5 MoE stage; training run.json required")
    identity(serve); serving(serve); hardware(serve)
    serve.add_argument("--stage", type=int, required=True); serve.add_argument("--base-path")
    serve.add_argument("--adapter-path"); serve.add_argument("--max-context", type=int, default=4096)
    serve.add_argument("--max-cache-mib", type=int, default=512); serve.set_defaults(function=_split_serve)
    split = sub.add_parser("split-chat", help="Generate with ordered cached MoE stages over signed RPC")
    identity(split); split.add_argument("--endpoints", required=True); split.add_argument("--base-path")
    split.add_argument("--allow-insecure-http", action="store_true"); split.add_argument("--raw-prompt", action="store_true")
    split.add_argument("--prompt", required=True); split.add_argument("--max-new-tokens", type=int, default=64)
    split.add_argument("--timeout", type=float, default=120); split.add_argument("--max-context", type=int, default=4096)
    split.set_defaults(function=_split_chat)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.function(args)
