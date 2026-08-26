# Installing SGLang on ROCm / RDNA3

A working recipe, and the reasoning behind each unusual step. Every command
here was run on the reference machine: 2× Radeon PRO W7900 (gfx1100, 48 GB),
ROCm 7.2.1, PyTorch 2.9.1+rocm, Python 3.12, Ubuntu 24.04.

## The core problem: pip will replace your ROCm PyTorch

SGLang's `pyproject.toml` lists CUDA-only packages as unconditional core
dependencies — no `sys_platform` or extra markers:

```
cuda-python>=13.0
cuda-tile==1.6.0rc5
flashinfer_python[cu13]==0.6.17
nvidia-cutlass-dsl[cu13]==4.6.2
nvidia-mathdx==25.6.0
```

A plain `pip install -e .` resolves these, and `flashinfer_python` requires
`torch`, so pip downloads a 527 MB CUDA build of PyTorch and installs it over
your ROCm one. The failure is silent until the next time you touch a GPU.

Filtering package names by hand is not enough: `humming-kernels[cu13]` drags
in the same CUDA torch transitively, and it is easy to miss. Constraining
torch with `-c constraints.txt` does not work either — pip gives up with
`ResolutionImpossible`.

**The reliable approach is `--no-deps` everywhere**, so pip's resolver never
runs, and then adding back what is genuinely needed.

## Step 1 — verify your starting point

```bash
python -c "import torch; print(torch.__version__, torch.version.hip)"
# 2.9.1+gitff65f5b 7.2.53211-e1a6bc5663      <- must show a hip version
rocm-smi --showid | grep -i gfx
# GFX Version: gfx1100
```

If `torch.version.hip` is `None`, you have a CUDA build; fix that first.

## Step 2 — get the SGLang source

```bash
git clone --depth 1 https://github.com/sgl-project/sglang.git
```

Behind a slow link, `--filter=blob:none` helps.

## Step 3 — install SGLang without its dependency tree

```bash
cd sglang/python
pip install --no-deps --no-build-isolation -e .
```

`--no-build-isolation` avoids a `BackendUnavailable: Cannot import
'wheel_stub.buildapi'` error.

## Step 4 — add the real dependencies

Take the dependency list, drop the CUDA and torch entries, and install the
rest with `--no-deps`:

```python
# save as filter_deps.py, run from sglang/python
import re, tomllib
BAD = {
    "cuda-python", "cuda-tile", "flashinfer-python", "flashinfer_python",
    "nvidia-cutlass-dsl", "nvidia-mathdx", "nvidia-ml-py", "nvidia-modelopt",
    "humming-kernels", "torch", "torchvision", "torchaudio", "triton",
    # Critical: this is the CUDA-only wheel. If pip installs it, it silently
    # replaces the gfx1100 kernel you build in step 6 and model registration
    # collapses from 249 architectures to 4.
    "sglang-kernel",
}
def name(s): return re.split(r"[\[<>=!;\s]", s.strip(), 1)[0].lower()
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
keep = [d for d in deps if name(d) not in BAD]
# strip [cu13]-style extras that pull CUDA transitives
print("\n".join(re.sub(r"\[cu\d+\]", "", d) for d in keep))
```

```bash
python filter_deps.py > deps.txt
pip install --no-deps --no-build-isolation -r deps.txt
```

Because `--no-deps` skips second-order requirements, a few need adding by
hand:

```bash
pip install --no-deps "huggingface_hub>=1.0,<2.0"  # transformers 5.x needs it
pip install --no-deps apache-tvm-ffi==0.1.11       # xgrammar needs it; not CUDA
pip install --no-deps kernels_data                 # sgl-kernel build needs it
```

Confirm nothing clobbered torch:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

## Step 5 — install this plugin

```bash
pip install --no-deps sglang-radeon-rdna3
```

## Step 6 — build sgl-kernel for gfx1100

SGLang's HIP path imports `sgl_kernel` unconditionally from
`layers/activation.py`, and that module is on the import path of every model —
so without it almost no model architecture registers. The PyPI wheel
(`sglang-kernel`) is CUDA-only: it looks for `libnvrtc.so.13` and misreads
gfx1100's `major=11, minor=0` as "CUDA compute capability 110 / SM110".

```bash
pip uninstall -y sglang-kernel        # if a CUDA build slipped in
sglang-rdna3 build-kernel --sglang-src /path/to/sglang --install
```

Two minutes on 32 cores. To inspect the patch first, add `--patch-only`; the
original is kept as `setup_rocm.py.orig`.

Verify the binary really targets your GPU:

```bash
strings $(python -c "import sgl_kernel,os;print(os.path.dirname(sgl_kernel.__file__))")/common_ops*.so \
  | grep -oE "gfx[0-9]+" | sort -u
# gfx1100
```

## Step 7 — check and serve

```bash
sglang-rdna3 doctor
```

You want `249 model architectures registered` (or similar). A single-digit
count means the import chain is broken — almost always `sgl_kernel`.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --tp-size 2 --attention-backend triton --disable-cuda-graph \
  --dtype bfloat16 --mem-fraction-static 0.85 --port 30000
```

## Notes for users in mainland China

`huggingface.co` is usually unreachable; the mirror works and is fast:

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install --no-deps hf_transfer hf_xet
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Qwen3.8-27B (54 GB, 18 shards) downloads in about three minutes this way.
`pip` benefits from a domestic index such as
`https://pypi.tuna.tsinghua.edu.cn/simple`.

## Troubleshooting

## `No module named sglang.launch_server`

If you cloned SGLang into your working directory, Python resolves
`import sglang` to that folder as a namespace package rather than to the
installed one. Everything installs cleanly and then:

```
/opt/venv/bin/python3: No module named sglang.launch_server
```

Diagnose it:

```bash
python3 -c "import sglang; print(sglang.__file__)"
```

* A path under `site-packages` or your source checkout's `python/` — fine.
* `AttributeError: module 'sglang' has no attribute '__file__'` — you have hit
  this. The import found a directory, not a package.

Fix: `cd` out of the directory containing `sglang/`, or clone somewhere that
is not your working directory. `install.sh` clones into `~/.sglang-rdna3` for
this reason.

The same shadowing catches you if a stray `sglang/` directory is left behind
by an interrupted download.

**`ModuleNotFoundError: No module named 'aiter.ops.triton.gemm'`** — the
plugin is not active. Check `sglang-rdna3 doctor`; if the shims show as not
applied, the `.pth` file did not install. Copy it manually:

```bash
python - <<'EOF'
import site, pathlib
pathlib.Path(site.getsitepackages()[0], "sglang_radeon_rdna3.pth").write_text(
    "import sglang_radeon_rdna3._bootstrap; "
    "sglang_radeon_rdna3._bootstrap.bootstrap()\n")
EOF
```

Errors in a `.pth` are printed by `site` and then ignored, so the symptom is
"nothing happened". Surface them with
`python -X importtime -c pass 2>&1 | grep radeon`.

**`TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were
given`** — same cause: the plugin is not loaded.

**Server starts, then a GPU memory access fault on the first request** — you
are serving an MoE model. See
[triton#10808](https://github.com/triton-lang/triton/issues/10808); use a
dense model until that lands.

**`-mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc`** —
harmless. aiter tries to JIT a custom all-reduce kernel with a CDNA-only flag,
fails, and SGLang falls back to NCCL.

**Everything hangs, `rocm-smi` included, in uninterruptible `D` state** — not
software. Suspect the GPU or its driver on that host and try another machine.
We hit exactly this on one node of the test cluster.

## Networking notes (China and other restricted networks)

GitHub reachability is not uniform, and not even consistent between hosts on
one network. On our test cluster one node could reach `github.com` while
another timed out on it but answered on `codeload.github.com`. So:

* **`install.sh` probes and falls back.** If `github.com` is unreachable it
  tries `ghproxy.net`, `gh-proxy.com`, `ghfast.top` in order.
  `GITHUB_MIRROR=https://your.proxy` pins one; `GITHUB_MIRROR=none` forces
  direct.
* **Mirrors are unreliable for large fetches.** A `git clone` of SGLang
  through `ghproxy.net` managed 2.8 MB in two minutes, and tarball requests
  timed out without a status code. Direct `codeload.github.com` did 3.6 MB in
  30 seconds. The script uses a tarball rather than a clone for exactly this
  reason — no git history to drag across.
* **PyPI mirrors work well.** Set `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.
* **Hugging Face needs a mirror.** `huggingface.co` times out; set
  `HF_ENDPOINT=https://hf-mirror.com`. With `hf_transfer` enabled, Qwen3.8-27B
  (54 GB, 18 shards) downloads in about three minutes.

If nothing works, clone on a machine that can reach GitHub, copy the tree
over, and point the installer at it:

```bash
SGLANG_SRC=/path/to/sglang bash install.sh
```

### Why not just install SGLang from PyPI?

It would be much faster — the wheel is 23 MB and lands in under a minute even
through a domestic mirror, versus minutes for the sources. But the wheel does
not ship the AoT kernel tree: no `setup_rocm.py`, no `csrc/gemm/gptq`, no
`common_extension_rocm.cc`. Without those there is no way to build for
gfx1100, and without that kernel SGLang registers about 4 model architectures
instead of 249. So the sources are not optional, and installing the Python
package from the same tarball keeps the two in step.
