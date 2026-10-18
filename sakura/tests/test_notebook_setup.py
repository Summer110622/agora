"""Exercise the shipped Colab cells without pip, network, GPU, or model downloads.

Each case uses an isolated, site-disabled Python process. An editable-install .pth
is created after interpreter startup, exactly when a live notebook would otherwise
miss it. The actual notebook cells must discover their checked checkout themselves.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


_ISOLATED_RUNNER = r'''
import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import site
import subprocess
import sys
from unittest import mock

configuration = json.loads(sys.argv[1])
repo = Path(configuration["repo"])
source = (repo / "sakura" / "src").resolve()
expected_module = source / "sakura" / "inference_generate.py"
runtime_site = Path(configuration["runtime_site"])
case = configuration["case"]
notebook = json.loads(Path(configuration["notebook"]).read_text(encoding="utf-8"))
cells = {cell["id"]: "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
         for cell in notebook["cells"] if cell["cell_type"] == "code"}

# A pip subprocess has written the editable path, but this already-running
# interpreter has not reprocessed its site files. Invalidate_caches alone also
# does not execute newly created .pth files.
runtime_site.mkdir()
(runtime_site / "sakura-editable.pth").write_text(str(source) + "\n", encoding="utf-8")
assert str(source) not in sys.path
assert importlib.util.find_spec("sakura") is None
importlib.invalidate_caches()
assert importlib.util.find_spec("sakura") is None
try:
    importlib.import_module("sakura")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Control import should fail before site/bootstrap discovery")

namespace = {"__name__": "__notebook__"}
exec(compile(cells["parameters"], "notebook-parameters", "exec"), namespace)
namespace.update(REPO_DIRECTORY=str(repo), DEVICE="cpu", MODE="local")
captured = io.StringIO()
pip_calls = []

def fake_git(command, **kwargs):
    if command[:3] != ["git", "-C", str(repo)]:
        raise AssertionError("Unexpected subprocess check_output: " + repr(command))
    if command[3:] == ["branch", "--show-current"]:
        return namespace["REPO_REF"] + "\n"
    if command[3:] == ["rev-parse", "HEAD"]:
        return "a" * 40 + "\n"
    raise AssertionError("Unexpected git query: " + repr(command))

def fake_pip(command, **kwargs):
    if command[:4] != [sys.executable, "-m", "pip", "install"]:
        raise AssertionError("Only pip installation may be simulated: " + repr(command))
    pip_calls.append(command)
    # A second editable install still cannot mutate the parent process sys.path.
    (runtime_site / "sakura-editable.pth").write_text(str(source) + "\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, 0)

def execute(cell):
    with contextlib.redirect_stdout(captured):
        exec(compile(cells[cell], "notebook-" + cell, "exec"), namespace)

if case == "restart":
    # Processing this site directory represents a fresh kernel discovering the
    # editable install. It must resolve the same package as the live bootstrap.
    site.addsitedir(str(runtime_site))
    assert Path(importlib.import_module("sakura").__file__).resolve() == source / "sakura" / "__init__.py"
elif case in {"unloaded-decoy", "cached-decoy"}:
    sys.path.insert(0, configuration["decoy_source"])
    if case == "cached-decoy":
        assert Path(importlib.import_module("sakura").__file__).resolve().is_relative_to(Path(configuration["decoy_source"]))
elif case == "namespace":
    # Running from a checkout root can cache the outer sakura/ directory as a
    # namespace package; its real package lives deeper at sakura/src/sakura.
    sys.path.insert(0, str(repo))
    assert getattr(importlib.import_module("sakura"), "__file__", None) is None
elif case == "missing-checkout":
    namespace["REPO_DIRECTORY"] = str(repo / "does-not-exist")

expected_error = case in {"cached-decoy", "namespace", "missing-checkout"}
failure = None
with mock.patch("importlib.metadata.version", return_value="2.10.0"), \
     mock.patch("subprocess.check_output", side_effect=fake_git), \
     mock.patch("subprocess.run", side_effect=fake_pip):
    try:
        if case in {"install", "repeat-install"}:
            execute("install")
            if case == "repeat-install":
                execute("install")
        else:
            # The parameter cell ran, but setup's repo variable never existed.
            assert "repo" not in namespace
        execute("load-local")
    except RuntimeError as error:
        if not expected_error:
            raise
        failure = str(error)

if expected_error:
    assert failure, "A missing or conflicting checkout needs an actionable RuntimeError"
    assert "sakura.inference_generate" not in sys.modules, "Conflicting package must not load a generator"
else:
    assert failure is None
    loaded = importlib.import_module("sakura.inference_generate")
    assert Path(loaded.__file__).resolve() == expected_module
    assert Path(namespace["generator"].loaded_from).resolve() == expected_module
    assert namespace["generator"].model_id == namespace["MODEL_ID"]
    assert namespace["generator"].revision == namespace["MODEL_REVISION"]
    assert sys.path.count(str(source)) == 1, "Rerunning cells must not grow sys.path"

print(json.dumps({"case": case, "expected_error": expected_error, "failure": failure,
                  "source_path_occurrences": sys.path.count(str(source)), "pip_calls": len(pip_calls)}, ensure_ascii=True))
'''


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "checked-repo"
        self.source = self.repo / "sakura" / "src"
        package = self.source / "sakura"
        package.mkdir(parents=True)
        (self.repo / "sakura_inference.py").write_text("# local launcher fixture\n", encoding="utf-8")
        (package / "__init__.py").write_text("__version__ = 'notebook-fixture'\n", encoding="utf-8")
        (package / "inference_generate.py").write_text(textwrap.dedent("""\
            class HFGenerator:
                @classmethod
                def from_pretrained(cls, model_id, revision, **kwargs):
                    instance = cls()
                    instance.loaded_from = __file__
                    instance.model_id = model_id
                    instance.revision = revision
                    return instance
            """), encoding="utf-8")
        self.decoy_source = self.root / "other-site"
        decoy = self.decoy_source / "sakura"
        decoy.mkdir(parents=True)
        (decoy / "__init__.py").write_text("__version__ = 'wrong-checkout'\n", encoding="utf-8")
        self.notebook = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pc_inference.ipynb"

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, case):
        configuration = {"case": case, "repo": str(self.repo), "decoy_source": str(self.decoy_source),
                         "runtime_site": str(self.root / "runtime-site"), "notebook": str(self.notebook)}
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", _ISOLATED_RUNNER,
                                 json.dumps(configuration)], cwd=self.root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_live_kernel_finds_checkout_after_editable_install_creates_pth(self):
        result = self.run_case("install")
        self.assertEqual(result["pip_calls"], 2)
        self.assertEqual(result["source_path_occurrences"], 1)

    def test_model_cell_recovers_without_setup_repo_variable(self):
        result = self.run_case("load-only")
        self.assertEqual(result["pip_calls"], 0)

    def test_restarted_kernel_uses_same_editable_checkout(self):
        self.run_case("restart")

    def test_repeating_setup_does_not_duplicate_source_search_path(self):
        result = self.run_case("repeat-install")
        self.assertEqual(result["pip_calls"], 4)
        self.assertEqual(result["source_path_occurrences"], 1)

    def test_source_path_takes_priority_over_unloaded_unrelated_package(self):
        self.run_case("unloaded-decoy")

    def test_cached_outer_namespace_requests_fresh_session(self):
        self.assertTrue(self.run_case("namespace")["expected_error"])

    def test_cached_other_package_is_rejected_before_loading_generator(self):
        self.assertTrue(self.run_case("cached-decoy")["expected_error"])

    def test_missing_checkout_reports_setup_error_before_import(self):
        self.assertTrue(self.run_case("missing-checkout")["expected_error"])


if __name__ == "__main__":
    unittest.main()
