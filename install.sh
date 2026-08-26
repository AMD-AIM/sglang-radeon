#!/usr/bin/env bash
# Install SGLang for AMD Radeon RDNA3 (gfx110x).
#
#   curl -fsSL https://raw.githubusercontent.com/AMD-AIM/sglang-radeon/main/install.sh | bash
#
# Assumes a working ROCm PyTorch. Everything else -- SGLang itself, the
# gfx1100 kernel, this plugin -- is handled here. See --help for options.
set -euo pipefail

REPO_SLUG="AMD-AIM/sglang-radeon"
SGLANG_SLUG="sgl-project/sglang"

# GitHub is unreachable from some networks (notably mainland China), and the
# reachable set is not consistent even between hosts on one cluster -- we hit
# a node where github.com timed out but codeload.github.com answered. So we
# probe rather than assume, and fall back to a mirror.
#   GITHUB_MIRROR=https://your.proxy   pins one explicitly
#   GITHUB_MIRROR=none                 forces direct
GITHUB_MIRROR="${GITHUB_MIRROR:-}"
MIRROR_CANDIDATES="https://ghproxy.net https://gh-proxy.com https://ghfast.top"
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
command -v curl >/dev/null || die "curl not found"

reachable() { curl -fsS -m 8 -o /dev/null "$1" 2>/dev/null; }

# Probe codeload, not github.com. We fetch sources as tarballs, and those
# come from codeload.github.com -- which on several networks answers when
# github.com itself does not. Testing the wrong host sends us to a mirror
# unnecessarily, or fails outright when the mirrors are blocked too.
if [ -z "$GITHUB_MIRROR" ]; then
  if reachable "https://codeload.github.com/$REPO_SLUG/tar.gz/refs/heads/main"; then
    GITHUB_MIRROR=none
  else
    for m in $MIRROR_CANDIDATES; do
      if reachable "$m"; then GITHUB_MIRROR="$m"; break; fi
    done
    [ -z "$GITHUB_MIRROR" ] && die "cannot reach codeload.github.com or any known mirror.
  Options:
    - set GITHUB_MIRROR=https://your.proxy
    - fetch the sources elsewhere and pass SGLANG_SRC=/path/to/sglang
    - pip download --no-deps --no-binary :all: sglang-radeon-rdna3
      (the sdist carries this installer and its helpers)"
  fi
fi
if [ "$GITHUB_MIRROR" = "none" ]; then
  ok "codeload.github.com reachable"
  tar_url() { printf 'https://codeload.github.com/%s/tar.gz/refs/heads/%s' "$1" "$2"; }
else
  ok "using mirror $GITHUB_MIRROR"
  tar_url() { printf '%s/https://github.com/%s/archive/refs/heads/%s.tar.gz' "$GITHUB_MIRROR" "$1" "$2"; }
fi

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
step "Installing SGLang"
# --------------------------------------------------------------------------
mkdir -p "$WORKDIR"

# We need SGLang's sources, not just the package: the published wheel omits
# the AoT kernel tree (no setup_rocm.py, no csrc/gemm/gptq), and without that
# there is no way to build for gfx1100. A tarball rather than a clone --
# same content, none of the ~500 MB of history, which matters a lot on a
# throttled link.
#
# --no-deps throughout: SGLang's dependency list hard-codes CUDA packages
# with no platform markers, and resolving them installs a CUDA build of
# PyTorch over the ROCm one.

if [ -n "${SGLANG_SRC:-}" ]; then
  SGLANG_DIR="$SGLANG_SRC"
  ok "using existing checkout at $SGLANG_DIR"
  pip install --no-deps --no-build-isolation -q -e "$SGLANG_DIR/python"
else
  SGLANG_DIR="$WORKDIR/sglang-src"
  if [ ! -f "$SGLANG_DIR/python/sglang/kernels/aot/setup_rocm.py" ]; then
    rm -rf "$SGLANG_DIR"; mkdir -p "$SGLANG_DIR"
    TARBALL="$WORKDIR/sglang-src.tar.gz"
    TAR_URL="$(tar_url "$SGLANG_SLUG" "$SGLANG_REF")"
    echo "  fetching kernel sources (a few minutes on a slow link)..."
    curl -fsSL --retry 3 -o "$TARBALL" "$TAR_URL" \
      || die "could not download SGLang sources from $TAR_URL"
    tar xzf "$TARBALL" -C "$SGLANG_DIR" --strip-components=1
    rm -f "$TARBALL"
  fi
  ok "SGLang sources in $SGLANG_DIR"

  # Install from the tree we just unpacked, not from PyPI. The wheel is much
  # faster to fetch, but the kernel we build has to match the Python code it
  # is loaded by -- and pinning both to one tarball is the only way to be
  # sure of that. (The AoT sources are absent from the wheel anyway, so the
  # tarball is not optional.)
  pip install --no-deps --no-build-isolation -q -e "$SGLANG_DIR/python"
  ok "sglang (editable, from $SGLANG_REF)"
fi

# The plugin. We still want the repo for scripts/filter_deps.py and the
# docs, but PyPI is a faster and more reliable source for the package itself
# on networks where GitHub is slow.
# A tarball, not a clone: git needs github.com, which is blocked on more
# networks than codeload is. PyPI's sdist is the second fallback -- it
# carries this installer and scripts/ as well as the package.
PLUGIN_DIR="$WORKDIR/plugin"
PLUGIN_SOURCE=""
if [ ! -f "$PLUGIN_DIR/scripts/filter_deps.py" ]; then
  rm -rf "$PLUGIN_DIR"; mkdir -p "$PLUGIN_DIR"
  if curl -fsSL --retry 2 -o "$WORKDIR/plugin.tar.gz" \
        "$(tar_url "$REPO_SLUG" main)" 2>/dev/null; then
    tar xzf "$WORKDIR/plugin.tar.gz" -C "$PLUGIN_DIR" --strip-components=1
    rm -f "$WORKDIR/plugin.tar.gz"
    PLUGIN_SOURCE="github"
  elif pip download --no-deps --no-binary :all: -d "$WORKDIR/sdist" \
          sglang-radeon-rdna3 -q 2>/dev/null; then
    tar xzf "$WORKDIR"/sdist/sglang_radeon_rdna3-*.tar.gz \
        -C "$PLUGIN_DIR" --strip-components=1
    PLUGIN_SOURCE="pypi"
  else
    die "could not obtain the plugin from GitHub or PyPI"
  fi
else
  PLUGIN_SOURCE="cached"
fi
ok "plugin sources ($PLUGIN_SOURCE)"

# --------------------------------------------------------------------------
step "Installing dependencies"
# --------------------------------------------------------------------------
DIFFUSION_FLAG=""
[ "$WITH_DIFFUSION" = "1" ] && DIFFUSION_FLAG="--diffusion"
python3 "$PLUGIN_DIR/scripts/filter_deps.py" \
    --pyproject "$SGLANG_DIR/python/pyproject.toml" \
    --output "$WORKDIR/deps.txt" $DIFFUSION_FLAG
pip install --no-deps --no-build-isolation -q -r "$WORKDIR/deps.txt"

# --no-deps skips second-order requirements; these are needed and are not
# CUDA-specific.
pip install --no-deps -q "huggingface_hub>=1.0,<2.0" apache-tvm-ffi==0.1.11 kernels_data
ok "dependencies installed"

# Verify nothing swapped torch out from under us.
NOW=$(python3 -c 'import torch; print(torch.version.hip or "NOHIP")' 2>/dev/null || echo NOHIP)
[ "$NOW" = "NOHIP" ] && die "a dependency replaced ROCm PyTorch with a CUDA build.
  Please report this along with the output above."
ok "ROCm PyTorch intact"

# --------------------------------------------------------------------------
step "Installing the RDNA3 plugin"
# --------------------------------------------------------------------------
pip install --no-deps -q "$PLUGIN_DIR"
ok "sglang-radeon-rdna3"

# --------------------------------------------------------------------------
if [ "$SKIP_KERNEL" = "0" ]; then
step "Building sgl-kernel for $GFX (a few minutes)"
# --------------------------------------------------------------------------
  # The PyPI 'sglang-kernel' wheel is CUDA-only and would shadow our build.
  pip uninstall -y -q sglang-kernel 2>/dev/null || true
  sglang-rdna3 build-kernel --sglang-src "$SGLANG_DIR" --jobs "$JOBS" --install
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
    $PLUGIN_DIR/docs/qwen3.8-27b.md
    $PLUGIN_DIR/docs/minimax-h3.md
EOF
else
  die "doctor reported problems — see above"
fi
