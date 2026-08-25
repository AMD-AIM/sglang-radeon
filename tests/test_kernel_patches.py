"""Tests for the sgl-kernel source patches.

The patcher edits C++ and build files by string anchor, so the risks are:
applying twice, misreporting an unpatched file as done, and silently doing
nothing when upstream moves. All three are covered here.
"""

import pytest

from sglang_radeon_rdna3 import kernel_patches as kp


@pytest.fixture
def aot(tmp_path):
    """A minimal stand-in for the AoT build tree."""
    (tmp_path / "csrc" / "gemm" / "gptq").mkdir(parents=True)

    (tmp_path / "setup_rocm.py").write_text(
        'sources = [\n'
        '    "csrc/elementwise/pos_enc.cu",\n'
        ']\n'
        'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:\n'
        '    sys.exit(1)\n'
        'topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target == "gfx942" '
        'else 32 * 1024 * 4\n'
    )
    (tmp_path / "csrc" / "common_extension_rocm.cc").write_text(
        '  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");\n'
    )
    (tmp_path / "csrc" / "gemm" / "gptq" / "gptq_kernel.cu").write_text(
        "#ifndef USE_ROCM\n"
        "      res2 = {};\n"
        "#else\n"
        "      res2.x = __half_as_ushort(__float2half(0));\n"
        "      res2.y = __half_as_ushort(__float2half(0));\n"
        "#endif\n"
        "#ifndef USE_ROCM\n"
        "      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));\n"
        "#else\n"
        "      res[m] = __hadd(res[m], __hadd(__ushort_as_half(res2.x), "
        "__ushort_as_half(res2.y)));\n"
        "#endif\n"
    )
    return tmp_path


ALL = kp.BASE_PATCHES + kp.GPTQ_PATCHES


def test_all_patches_apply_cleanly(aot):
    results = kp.apply(aot, ALL)
    assert all("applied" in r for r in results), results
    assert not any("ANCHOR NOT FOUND" in r for r in results), results

    setup = (aot / "setup_rocm.py").read_text()
    assert "gfx1100" in setup
    assert "_LDS_64KB_ARCHS" in setup
    assert "gptq_kernel.cu" in setup

    ext = (aot / "csrc" / "common_extension_rocm.cc").read_text()
    assert "gptq_shuffle" in ext and "gptq_gemm" in ext

    # The stale ROCm branches must be gone, not merely bypassed.
    kernel = (aot / "csrc" / "gemm" / "gptq" / "gptq_kernel.cu").read_text()
    assert "__ushort_as_half" not in kernel
    assert "__half_as_ushort" not in kernel
    # Brace-init is ambiguous for HIP __half2 and must not be reintroduced.
    assert "res2 = {}" not in kernel
    assert "__halves2half2" in kernel


def test_patches_are_idempotent(aot):
    kp.apply(aot, ALL)
    after_first = (aot / "csrc" / "gemm" / "gptq" / "gptq_kernel.cu").read_text()

    results = kp.apply(aot, ALL)
    assert all("already applied" in r for r in results), results
    assert (aot / "csrc" / "gemm" / "gptq" / "gptq_kernel.cu").read_text() == after_first


def test_backup_is_written_once(aot):
    kp.apply(aot, ALL)
    backup = aot / "setup_rocm.py.orig"
    assert backup.exists()
    assert "gfx1100" not in backup.read_text()  # pristine

    original = backup.read_text()
    kp.apply(aot, ALL)
    assert backup.read_text() == original  # not overwritten by the second run


def test_missing_anchor_is_reported_not_silent(aot):
    (aot / "setup_rocm.py").write_text("# upstream rewrote this file\n")
    results = kp.apply(aot, kp.BASE_PATCHES)
    assert any("ANCHOR NOT FOUND" in r for r in results), results


def test_missing_file_is_reported(tmp_path):
    results = kp.apply(tmp_path, kp.BASE_PATCHES)
    assert any("MISSING" in r for r in results), results


def test_base_patches_alone_do_not_touch_gptq(aot):
    kp.apply(aot, kp.BASE_PATCHES)
    setup = (aot / "setup_rocm.py").read_text()
    assert "gfx1100" in setup
    assert "gptq_kernel.cu" not in setup
