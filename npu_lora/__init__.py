"""Deprecated alias of `npu_train` (the package was renamed from `npu_lora`).

`import npu_lora.X` returns the very module object `npu_train.X` (one copy of every class and global), and
`python -m npu_lora.X ...` runs `npu_train.X` as `__main__`, so existing commands keep working. A file that
exists only under this directory (for example a script copied here on a host) is still found here.

Use `npu_train` in new code. This alias is removed in a later release.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings

warnings.warn("the package npu_lora was renamed to npu_train; `import npu_lora` and `python -m npu_lora.<module>` "
              "still work but are deprecated", DeprecationWarning, stacklevel=2)

_OLD, _NEW = "npu_lora", "npu_train"

try:
    import npu_train as _pkg
except ImportError:
    # An editable install made before the rename maps only `npu_lora` to its directory; npu_train sits next to it.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    import npu_train as _pkg

__version__ = _pkg.__version__


def __getattr__(name):
    # Top-level names (LoRATrainer, ... are lazy in npu_train) come from npu_train.
    return getattr(_pkg, name)


class _AliasLoader(importlib.abc.Loader):
    """Loads `npu_lora.X` as the already-importable module `npu_train.X`."""

    def __init__(self, real_name, real_spec):
        self.real_name, self.real_spec = real_name, real_spec

    def create_module(self, spec):
        mod = importlib.import_module(self.real_name)
        self._real_module_spec = mod.__spec__
        return mod

    def exec_module(self, module):
        # The import system sets module.__spec__ to the alias spec; keep the module's own (pickle, reload, spawn).
        module.__spec__ = self._real_module_spec

    # runpy (`python -m npu_lora.X`) asks the loader for the code and runs it as __main__.
    def get_code(self, fullname):
        return self.real_spec.loader.get_code(self.real_name)

    def get_source(self, fullname):
        return self.real_spec.loader.get_source(self.real_name)

    def is_package(self, fullname):
        return self.real_spec.submodule_search_locations is not None

    def get_filename(self, fullname):
        return self.real_spec.origin


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real_name = _NEW + fullname[len(_OLD):]
        try:
            real = importlib.util.find_spec(real_name)
        except (ImportError, ValueError):
            real = None
        if real is None:
            return None  # not in npu_train: fall through to files under npu_lora/
        spec = importlib.machinery.ModuleSpec(fullname, _AliasLoader(real_name, real), origin=real.origin,
                                              is_package=real.submodule_search_locations is not None)
        if real.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real.submodule_search_locations)
        return spec


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
