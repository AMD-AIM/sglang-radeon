"""Ensure the compatibility shims run before SGLang's own modules import.

The timing problem
------------------
SGLang exposes an entry-point plugin hook, but it runs inside
``entrypoints/engine.py`` and ``managers/scheduler.py`` -- after ``import
sglang`` has already completed, and the diffusion server never calls it at
all. The aiter defect we work around fires during module import, well before
that:

    scheduler.py:70
      -> configs/model_config.py:31
        -> layers/quantization/__init__.py:57
          -> quark/quark.py:20
            -> quark/schemes/__init__.py:4
              -> quark/schemes/quark_w4a4_mxfp4.py:32
                 from aiter.ops.triton.gemm.fused... import ...
                 ModuleNotFoundError

So the entry point alone cannot save us. This module is executed by a ``.pth``
file at interpreter startup (the ``site`` module runs those before any user
code), which is early enough.

It installs a lightweight :class:`importlib.abc.MetaPathFinder` rather than
applying the shims immediately: that keeps the cost near zero for Python
processes in this environment that never touch SGLang.

...and the opposite timing problem
----------------------------------
Being early is not free. The finder fires on the *first* ``sglang*`` import,
so the shims run while ``import sglang`` is still executing -- which means any
shim that needs to import an SGLang submodule of its own cannot, and reports
itself as "skipped (... not available)". That is how ``nested_weight_detection``
came to never apply in a real serve process, while ``sglang-rdna3 doctor`` --
which runs later, with everything already imported -- reported it as fine.

The fix is to stop treating the first pass as the only pass. Shims marked
deferrable in :mod:`~sglang_radeon_rdna3.compat` report "deferred" rather than
"skipped" when their target is not importable yet, and this finder stays
installed and retries them once ``import sglang`` has finished. It retires as
soon as nothing is deferred, so the steady-state cost is unchanged.
"""

from __future__ import annotations

import os
import sys
from importlib.abc import MetaPathFinder

_INSTALLED = False
_APPLIED_ONCE = False
_APPLYING = False

#: Backstop for an install where a deferred shim's target will never appear
#: (an LLM-only environment with no diffusion stack, say). Without it the
#: finder would keep retrying a doomed import for the life of the process.
_MAX_RETRIES = 20
_retries = 0


def _sglang_finished_importing() -> bool:
    """True once the top-level ``sglang`` package has finished executing.

    ``__spec__._initializing`` is the interpreter's own flag for "this module
    is part-way through its body" -- exactly the window in which a deferred
    shim cannot import what it needs.
    """
    module = sys.modules.get("sglang")
    if module is None:
        return False
    return not getattr(getattr(module, "__spec__", None), "_initializing", False)


class _SGLangImportWatcher(MetaPathFinder):
    """Applies the shims on the first SGLang import, then retries the rest."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            if not _APPLIED_ONCE or _sglang_finished_importing():
                _apply()
        return None  # never claim the module; let the real finders resolve it


def _retire() -> None:
    global _INSTALLED
    for finder in list(sys.meta_path):
        if isinstance(finder, _SGLangImportWatcher):
            sys.meta_path.remove(finder)
    _INSTALLED = False


def _apply() -> None:
    global _APPLIED_ONCE, _APPLYING, _retries

    # The shims import SGLang modules, which re-enters find_spec. Guard against
    # that rather than retiring first, because we may need to stay installed.
    if _APPLYING:
        return
    _APPLYING = True
    try:
        from sglang_radeon_rdna3 import compat

        compat.apply_all()
        if _APPLIED_ONCE:
            _retries += 1
        _APPLIED_ONCE = True
        if not compat.deferred_shims() or _retries >= _MAX_RETRIES:
            _retire()
    except Exception:
        if os.environ.get("SGLANG_RDNA3_DEBUG"):
            import traceback

            traceback.print_exc()
        _retire()  # something is wrong with the shims themselves; stop trying
    finally:
        _APPLYING = False


def bootstrap() -> None:
    """Idempotent; invoked from the .pth file and from the package __init__."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("SGLANG_RDNA3_ENABLE", "1").lower() in ("0", "false", "no"):
        return

    sys.meta_path.insert(0, _SGLangImportWatcher())
    _INSTALLED = True

    # Already past the risky window: apply directly (safe at any point).
    if any(m == "sglang" or m.startswith("sglang.") for m in sys.modules):
        _apply()
