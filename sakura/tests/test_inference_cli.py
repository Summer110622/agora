import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest import mock

import pytest

from sakura.auth import Identity
from sakura.inference_cli import load_inference_manifest, main, parser
from sakura.inference_generate import DEFAULT_MODEL, DEFAULT_REVISION


def test_init_keys_manifest_and_no_overwrite():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "inference"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["init", "--directory", str(directory)])
        manifest = load_inference_manifest(directory / "inference-run.json")
        assert manifest["model_id"] == DEFAULT_MODEL
        assert manifest["revision"] == DEFAULT_REVISION
        for member in ("pc", "colab"):
            identity = Identity.load(directory / f"{member}.private.json")
            assert manifest["members"][member] == identity.public_key
        assert "private_key" not in (directory / "inference-run.json").read_text()
        assert "private_key" not in output.getvalue()
        originals = {p.name: p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(FileExistsError):
            main(["init", "--directory", str(directory)])
        assert originals == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_init_bad_revision_or_context_has_no_filesystem_side_effect():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "bad"
        for options in (["--revision", "main"], ["--max-context", "1"], ["--model", "different/model"]):
            with pytest.raises(ValueError):
                main(["init", "--directory", str(directory), *options])
            assert not directory.exists()


def test_manifest_rejects_role_collision_and_training_manifest():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "inference"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--directory", str(directory)])
        path = directory / "inference-run.json"
        value = json.loads(path.read_text())
        value["roles"]["worker"] = "pc"
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="distinct"):
            load_inference_manifest(path)
        path.write_text('{"schema_version": 1, "mode": "lora"}')
        with pytest.raises(ValueError, match="inference manifest"):
            load_inference_manifest(path)


def test_worker_identity_and_remote_plaintext_rejected_before_model_download():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "inference"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--directory", str(directory)])
        options = ["worker", "--manifest", str(directory / "inference-run.json"),
                   "--endpoint", "http://192.0.2.1:49201/rpc"]
        with mock.patch("sakura.inference_generate.HFGenerator.from_pretrained") as load:
            with pytest.raises(ValueError, match="worker private"):
                main([*options, "--identity", str(directory / "pc.private.json")])
            with pytest.raises(ValueError, match="HTTPS"):
                main([*options, "--identity", str(directory / "colab.private.json")])
            load.assert_not_called()


def test_cli_remote_generation_reports_entire_result():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "inference"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--directory", str(directory)])
        output = io.StringIO()
        result = {"text": "def add(a,b): return a+b", "metrics": {"generated_tokens": 8}, "end_to_end_seconds": 1}
        with mock.patch("sakura.inference_cli.request_generation", return_value=result) as generate:
            with contextlib.redirect_stdout(output):
                main(["chat", "--manifest", str(directory / "inference-run.json"),
                      "--identity", str(directory / "pc.private.json"), "--endpoint", "http://127.0.0.1:49201/rpc",
                      "--prompt", "Write add", "--max-new-tokens", "32"])
        assert json.loads(output.getvalue()) == result
        assert generate.call_args.args[1:] == ("Write add", 32)


def test_split_serve_supports_native_tls_and_cache_budget():
    args = parser().parse_args(["split-serve", "--manifest", "run.json", "--identity", "worker.private.json",
                               "--stage", "1", "--tls-cert", "cert.pem", "--tls-key", "key.pem",
                               "--max-cache-mib", "256", "--dtype", "float16"])
    assert (args.stage, args.tls_cert, args.max_cache_mib, args.dtype) == (1, "cert.pem", 256, "float16")
