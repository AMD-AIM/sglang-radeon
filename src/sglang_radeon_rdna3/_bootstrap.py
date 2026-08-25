"""Ensure the compatibility shims run before SGLang's own modules import.

The timing problem
------------------
SGLang exposes an entry-point plugin hook, but it runs inside
``entrypoints/engine.py`` and ``managers/scheduler.py`` -- after ``import
sglang`` has already completed. The aiter defect we work around fires during
module import, well before that:

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
processes in this environment that never touch SGLang. The finder fires on the
first ``sglang*`` import, applies the shims, and removes itself.
"""

from __future__ import annotations

import os
import sys
from importlib.abc import MetaPathFinder

_INSTALLED = False


class _SGLangImportWatcher(MetaPathFinder):
    """Applies the shims when SGLang is first imported, then retires."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            _apply_and_retire()
        return None  # never claim the module; let the real finders resolve it


def _apply_and_retire() -> None:
    global _INSTALLED
    for finder in list(sys.meta_path):
        if isinstance(finder, _SGLangImportWatcher):
            sys.meta_path.remove(finder)
    _INSTALLED = False
    try:
        from sglang_radeon_rdna3 import compat

        compat.apply_all()
    except Exception:
        if os.environ.get("SGLANG_RDNA3_DEBUG"):
            import traceback

            traceback.print_exc()


def bootstrap() -> None:
    """Idempotent; invoked from the .pth file and from the package __init__."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("SGLANG_RDNA3_ENABLE", "1").lower() in ("0", "false", "no"):
        return

    # Already past the risky window: apply directly (safe at any point).
    if any(m == "sglang" or m.startswith("sglang.") for m in sys.modules):
        _apply_and_retire()
        return

    sys.meta_path.insert(0, _SGLangImportWatcher())
    _INSTALLED = True
