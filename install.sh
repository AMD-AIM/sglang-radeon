#!/usr/bin/env bash
# Install SGLang for AMD Radeon RDNA3 (gfx110x).
#
#   curl -fsSL https://raw.githubusercontent.com/AMD-AIM/sglang-radeon/main/install.sh | bash
#
# Assumes a working ROCm PyTorch. Everything else -- SGLang itself, the
# gfx1100 kernel, this plugin -- is handled here. See --help for options.
set -euo pipefail

REPO_URL="${SGLANG_RDNA3_REPO:-https://github.com/AMD-AIM/sglang-radeon.git}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
SGLANG_REF="${SGLANG_REF:-main}"
WORKDIR="${SGLANG_RDNA3_WORKDIR:-$HOME/.sglang-rdna3}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
WITH_DIFFUSION=0
SKIP_KERNEL=0

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '\n%serror:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: install.sh [options]

  --with-diffusion   Also install the extras for MiniMax-H3 video generation
  --skip-kernel      Do not build sgl-kernel (you will have almost no models)
  --workdir DIR      Where to check out sources (default: ~/.sglang-rdna3)
  --jobs N           Parallel compile jobs (default: nproc)
  -h, --help         This message

Environment:
  SGLANG_REF         SGLang git ref to build against (default: main)
  HF_ENDPOINT        Set to https://hf-mirror.com behind the GFW
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-diffusion) WITH_DIFFUSION=1; shift ;;
    --skip-kernel)    SKIP_KERNEL=1; shift ;;
    --workdir)        WORKDIR="$2"; shift 2 ;;
    --jobs)           JOBS="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# --------------------------------------------------------------------------
step "Checking prerequisites"
# --------------------------------------------------------------------------
command -v python3 >/dev/null || die "python3 not found"
command -v git >/dev/null || die "git not found"

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PYV" in
  3.1[0-9]) ok "Python $PYV" ;;
  *) die "Python 3.10+ required, found $PYV" ;;
esac

# The single most important check: a CUDA build of torch here means pip has
# already clobbered the ROCm one, and nothing downstream will work.
TORCH_INFO=$(python3 - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.__version__, torch.version.hip or "NOHIP")
except Exception:
    print("MISSING", "MISSING")
PY
)
read -r TORCH_VER TORCH_HIP <<<"$TORCH_INFO"
[ "$TORCH_VER" = "MISSING" ] && die "PyTorch not installed. Install a ROCm build first:
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.0"
[ "$TORCH_HIP" = "NOHIP" ] && die "PyTorch $TORCH_VER is a CUDA build, not ROCm.
  Reinstall from https://download.pytorch.org/whl/rocmX.Y before continuing."
ok "PyTorch $TORCH_VER (HIP $TORCH_HIP)"

GFX=$(python3 - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
          if torch.cuda.is_available() else "NOGPU")
except Exception:
    print("NOGPU")
PY
)
case "$GFX" in
  gfx1100|gfx1101|gfx1102) ok "GPU: $GFX (RDNA3)" ;;
  NOGPU) die "No GPU visible to PyTorch. Check that /dev/kfd and /dev/dri are present." ;;
  *) warn "GPU is $GFX, not RDNA3 — continuing, but this package targets gfx110x" ;;
esac

# --------------------------------------------------------------------------
step "Fetching sources into $WORKDIR"
# --------------------------------------------------------------------------
mkdir -p "$WORKDIR"

clone_or_update() {  # $1=url $2=dir $3=ref
  if [ -d "$2/.git" ]; then
    git -C "$2" fetch --depth 1 origin "$3" -q && git -C "$2" checkout -q FETCH_HEAD
  else
    git clone --depth 1 --branch "$3" "$1" "$2" -q 2>/dev/null \
      || git clone --depth 1 "$1" "$2" -q
  fi
}

clone_or_update "$SGLANG_REPO" "$WORKDIR/sglang" "$SGLANG_REF"
ok "sglang at $(git -C "$WORKDIR/sglang" rev-parse --short HEAD)"
clone_or_update "$REPO_URL" "$WORKDIR/plugin" main
ok "plugin at $(git -C "$WORKDIR/plugin" rev-parse --short HEAD)"

# --------------------------------------------------------------------------
step "Installing SGLang (without its dependency resolver)"
# --------------------------------------------------------------------------
# SGLang's dependency list hard-codes CUDA packages with no platform markers
# (cuda-python, flashinfer[cu13], nvidia-cutlass-dsl, sglang-kernel...). Let
# pip resolve them and it installs a CUDA build of torch over your ROCm one.
# So: --no-deps everywhere, then add back what is genuinely needed.
pip install --no-deps --no-build-isolation -q -e "$WORKDIR/sglang/python"
ok "sglang (editable, no deps)"

DIFFUSION_FLAG=""
[ "$WITH_DIFFUSION" = "1" ] && DIFFUSION_FLAG="--diffusion"
python3 "$WORKDIR/plugin/scripts/filter_deps.py" \
    --pyproject "$WORKDIR/sglang/python/pyproject.toml" \
    --output "$WORKDIR/deps.txt" $DIFFUSION_FLAG
ok "$(wc -l < "$WORKDIR/deps.txt") dependencies kept, CUDA-only ones dropped"

pip install --no-deps --no-build-isolation -q -r "$WORKDIR/deps.txt"

# --no-deps skips second-order requirements; these three are needed and are
# not CUDA-specific.
pip install --no-deps -q "huggingface_hub>=1.0,<2.0" apache-tvm-ffi==0.1.11 kernels_data
ok "runtime dependencies"

# Verify nothing swapped torch out from under us.
NOW=$(python3 -c 'import torch; print(torch.version.hip or "NOHIP")' 2>/dev/null || echo NOHIP)
[ "$NOW" = "NOHIP" ] && die "a dependency replaced ROCm PyTorch with a CUDA build.
  Please report this with the output above."
ok "ROCm PyTorch intact"

# --------------------------------------------------------------------------
step "Installing the RDNA3 plugin"
# --------------------------------------------------------------------------
pip install --no-deps -q "$WORKDIR/plugin"
ok "sglang-radeon-rdna3"

# --------------------------------------------------------------------------
if [ "$SKIP_KERNEL" = "0" ]; then
step "Building sgl-kernel for $GFX (a few minutes)"
# --------------------------------------------------------------------------
  # The PyPI 'sglang-kernel' wheel is CUDA-only and would shadow our build.
  pip uninstall -y -q sglang-kernel 2>/dev/null || true
  sglang-rdna3 build-kernel --sglang-src "$WORKDIR/sglang" --jobs "$JOBS" --install
else
  warn "skipping kernel build: expect very few model architectures to register"
fi

# --------------------------------------------------------------------------
step "Verifying"
# --------------------------------------------------------------------------
if sglang-rdna3 doctor; then
  cat <<EOF

$GREEN$BOLD Ready. $OFF

  Serve a model:
    ${BOLD}sglang-rdna3 serve-args Qwen/Qwen3.8-27B${OFF}

  Docs:
    $WORKDIR/plugin/docs/qwen3.8-27b.md
    $WORKDIR/plugin/docs/minimax-h3.md
EOF
else
  die "doctor reported problems — see above"
fi
