"""Run SGLang on AMD Radeon RDNA3 (gfx110x) GPUs.

Installing this package is enough: it registers a ``sglang.srt.plugins``
entry point, so SGLang loads it automatically in both the engine process and
each scheduler subprocess. No SGLang source changes are required.

What it does:
  * applies compatibility shims for upstream defects that break RDNA3
    (see :mod:`sglang_radeon_rdna3.compat`);
  * exposes RDNA3-aware serving defaults (see
    :mod:`sglang_radeon_rdna3.hardware`);
  * ships ``sglang-rdna3``, a CLI to check the environment and build the
    gfx1100 sgl-kernel wheel.

Verified on Radeon PRO W7900 (gfx1100, 48 GB) with ROCm 7.2.1 / PyTorch 2.9.1.
"""

from __future__ import annotations

import logging
import os

from sglang_radeon_rdna3 import compat
from sglang_radeon_rdna3.hardware import (
    RDNA3_TARGETS,
    GpuInfo,
    detect_gpus,
    recommended_server_args,
)

__version__ = "0.1.0"

__all__ = [
    "GpuInfo",
    "RDNA3_TARGETS",
    "detect_gpus",
    "recommended_server_args",
    "register",
    "__version__",
]

logger = logging.getLogger(__name__)

#: Set to "0" to load the plugin but skip all patching (useful for A/B testing
#: whether a shim is still needed after an upstream release).
_ENABLE_ENV = "SGLANG_RDNA3_ENABLE"


def register() -> None:
    """Entry point invoked by SGLang's ``load_plugins()``.

    Kept deliberately cheap and failure-tolerant: a plugin that raises would
    take down the whole engine, so problems are logged rather than propagated.
    """
    if os.environ.get(_ENABLE_ENV, "1").lower() in ("0", "false", "no"):
        logger.info("sglang-radeon-rdna3 disabled via %s", _ENABLE_ENV)
        return

    status = compat.apply_all()
    applied = [k for k, v in status.items() if v.startswith("applied")]
    logger.info(
        "sglang-radeon-rdna3 %s loaded; shims applied: %s",
        __version__,
        ", ".join(applied) if applied else "none needed",
    )
