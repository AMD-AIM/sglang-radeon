"""Install hook that places our .pth file in the site-packages root.

All real metadata lives in pyproject.toml. The one thing that needs
imperative handling is the .pth file, because it is what makes the package
work with nothing more than `pip install`.

Why not `data_files`
--------------------
`data_files=[("", ["x.pth"])]` resolves against `sys.prefix`, so the file
lands in e.g. /opt/venv/x.pth rather than
/opt/venv/lib/python3.12/site-packages/x.pth. The `site` module only executes
.pth files found in site directories, so the former is silently inert --
the package installs cleanly and then does nothing.

Instead we mark the .pth as package data and copy it into the install
directory from a post-install hook, which works for both wheel installs and
`pip install -e .`.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

PTH_NAME = "sglang_radeon_rdna3.pth"
PTH_LINE = (
    "import sglang_radeon_rdna3._bootstrap; "
    "sglang_radeon_rdna3._bootstrap.bootstrap()\n"
)


def _write_pth(target_dir: str | None) -> None:
    if not target_dir:
        return
    try:
        Path(target_dir, PTH_NAME).write_text(PTH_LINE)
    except OSError as exc:  # pragma: no cover - non-fatal
        print(f"warning: could not install {PTH_NAME}: {exc}")
        print("         run `sglang-rdna3 doctor` to verify shim activation.")


class install(_install):
    def run(self):
        super().run()
        _write_pth(self.install_lib)


class develop(_develop):
    def run(self):
        super().run()
        _write_pth(self.install_dir)


class build_py(_build_py):
    """Ensure the .pth exists in the source tree for sdist/wheel builds."""

    def run(self):
        Path(__file__).parent.joinpath(PTH_NAME).write_text(PTH_LINE)
        super().run()


setup(cmdclass={"install": install, "develop": develop, "build_py": build_py})
