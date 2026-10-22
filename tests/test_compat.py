"""The old package name npu_lora keeps working after the rename to npu_train (issue #46).

Every check runs in a fresh interpreter so the one-time DeprecationWarning and the import state are not shared
with the other tests.
"""
import os
import shutil
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(args, cwd=ROOT, **kw):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300, **kw)


def test_import_old_name_is_same_module_and_warns():
    code = textwrap.dedent("""
        import sys, warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import npu_lora.metrics
        assert any(issubclass(x.category, DeprecationWarning) and "npu_train" in str(x.message) for x in w), w
        import npu_train, npu_train.metrics
        assert npu_lora.metrics is npu_train.metrics
        assert sys.modules["npu_lora.metrics"] is npu_train.metrics
        assert npu_train.metrics.__spec__.name == "npu_train.metrics"
        from npu_lora.k8s import build
        from npu_train.k8s import build as build2
        assert build is build2
        import npu_lora
        assert npu_lora.__version__ == npu_train.__version__
        assert npu_lora.LoRAConfig is npu_train.LoRAConfig        # lazy top-level names
        try:
            import npu_lora.no_such_module
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("missing module imported")
        print("ok")
    """)
    r = _run(["-c", code])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_python_m_old_name_runs():
    r = _run(["-m", "npu_lora.cli", "--help"])
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout and "train" in r.stdout
    r = _run(["-m", "npu_lora.compare", "--help"])
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


def _copy_tree(dst):
    for pkg in ("npu_train", "npu_lora"):
        shutil.copytree(os.path.join(ROOT, pkg), os.path.join(dst, pkg),
                        ignore=shutil.ignore_patterns("__pycache__"))


def test_script_left_in_old_directory_still_runs(tmp_path):
    # A script copied into npu_lora/ on a host (e.g. moe_bench2.py) imports through the alias and runs with -m.
    _copy_tree(tmp_path)
    (tmp_path / "npu_lora" / "local_tool.py").write_text(textwrap.dedent("""
        import sys
        from npu_lora.k8s import build
        from .compare import main
        import npu_train.k8s
        assert build is npu_train.k8s.build
        print("local", __name__, sys.argv[1:])
    """))
    # -S -E -s: stdlib only, so npu_train comes from the copied tree, not from an install
    r = _run(["-S", "-E", "-s", "-m", "npu_lora.local_tool", "x"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "local __main__ ['x']"


def test_old_editable_install_mapping_only_npu_lora(tmp_path):
    # An editable install made before the rename maps only `npu_lora` to its directory; npu_train is not on sys.path.
    _copy_tree(tmp_path / "repo")
    code = textwrap.dedent(f"""
        import importlib.util, sys
        class OldEditableFinder:
            @classmethod
            def find_spec(cls, name, path=None, target=None):
                if name == "npu_lora":
                    d = {str(tmp_path / "repo" / "npu_lora")!r}
                    return importlib.util.spec_from_file_location(name, d + "/__init__.py",
                                                                  submodule_search_locations=[d])
        sys.meta_path.append(OldEditableFinder)
        import npu_lora.k8s, npu_train.k8s
        assert npu_lora.k8s is npu_train.k8s
        print("ok")
    """)
    r = _run(["-S", "-E", "-s", "-c", code], cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
