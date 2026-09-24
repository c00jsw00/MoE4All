#!/usr/bin/env bash
# MoE4All — Linux build-toolchain installer (no sudo required).
#
# What this does
# --------------
# MoE4All compiles its Vulkan compute shaders with `glslc` at `cargo build` time.
# The shaders use `GL_KHR_cooperative_matrix`, so the bundled `glslc` must be
# recent enough. Many distros ship an old `shaderc`/`glslc` that fails on those
# shaders with:
#     error: '#extension' : extension not supported: GL_KHR_cooperative_matrix
#
# This script installs a known-good `glslc` (from the LunarG Vulkan SDK — the same
# reference toolchain the Windows build uses) into your home directory and puts a
# wrapper on your PATH. It is idempotent: if a suitable glslc is already available
# it leaves it alone.
#
# Robustness
# ----------
# * The SDK tarball is cached under $PREFIX/cache/moe4all-glslc and verified by a
#   full archive listing before use — a corrupt or truncated download is detected
#   and re-downloaded (with resume) instead of failing with a cryptic tar error.
# * Members are extracted by EXACT path (discovered from the archive listing),
#   never by glob, so behavior is identical across GNU tar / bsdtar.
# * Only glslc + the shared libraries it links (per ldd) are extracted.
#
# Usage
# -----
#   ./scripts/install-linux.sh                 # auto: reuse existing good glslc, else install SDK one
#   ./scripts/install-linux.sh --sdk-version 1.4.357.0
#   ./scripts/install-linux.sh --prefix ~/.local
#   ./scripts/install-linux.sh --force         # reinstall even if a good glslc is on PATH
#   MOE4ALL_SDK_TARBALL=/path/to/local.tar.xz ./scripts/install-linux.sh   # offline mirror
#
# After it succeeds, build MoE4All with:
#   cargo build --release --locked -p infr-cli
#   ./target/release/infr devices
#
set -euo pipefail

SDK_VERSION="${SDK_VERSION:-1.4.357.0}"
PREFIX="${PREFIX:-$HOME/.local}"
FORCE=0

# --- args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --sdk-version) SDK_VERSION="$2"; shift 2;;
    --sdk-version=*) SDK_VERSION="${1#*=}"; shift;;
    --prefix) PREFIX="$2"; shift 2;;
    --prefix=*) PREFIX="${1#*=}"; shift;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n '2,36p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
# The Vulkan SDK tarball layout is <ver>/x86_64/{bin,lib}. Standard LunarG host.
SDK_TARBALL_URL="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkan-sdk-${SDK_VERSION}.0.tar.xz"

# A shader that exercises GL_KHR_cooperative_matrix — the exact feature the
# MoE4All GEMM/attention shaders require. `#extension ... : require` fails on a
# glslc that predates KHR cooperative matrix, and succeeds (unused or not) on one
# that has it — so this is a clean version gate with no other extensions to drag in.
COOPMAT_PROBE='#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32) in;
void main() { }'

glslc_ok() {
  # $1 = candidate glslc executable
  local exe="$1"
  [ -x "$exe" ] || return 1
  local tmp
  tmp="$(mktemp -d)" || return 1
  printf '%s' "$COOPMAT_PROBE" > "$tmp/probe.comp"
  "$exe" -fshader-stage=comp --target-env=vulkan1.3 -O "$tmp/probe.comp" -o "$tmp/probe.spv" >/dev/null 2>&1
  local rc=$?
  rm -rf "$tmp"
  [ "$rc" -eq 0 ]
}

echo "==> MoE4All Linux glslc installer"
echo "    prefix        : $PREFIX"
echo "    sdk version   : $SDK_VERSION"

# 1) Is there already a suitable glslc?
if [ "$FORCE" -ne 1 ]; then
  existing="$(command -v glslc || true)"
  if [ -n "$existing" ]; then
    if glslc_ok "$existing"; then
      echo "==> Found a suitable glslc on PATH: $existing"
      "$existing" --version 2>/dev/null | head -1 || true
      echo "    Nothing to do. Build with: cargo build --release --locked -p infr-cli"
      exit 0
    else
      echo "==> Existing glslc ($existing) is too old (no GL_KHR_cooperative_matrix)."
      echo "    Installing a known-good one under $PREFIX."
    fi
  else
    echo "==> No glslc on PATH. Installing a known-good one under $PREFIX."
  fi
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }
command -v tar  >/dev/null 2>&1 || { echo "error: tar is required" >&2; exit 1; }
command -v ldd  >/dev/null 2>&1 || { echo "error: ldd is required (glibc)" >&2; exit 1; }

# 2) Obtain the SDK tarball: local override > verified cache > download (resumable).
cache_dir="$PREFIX/cache/moe4all-glslc"
mkdir -p "$cache_dir"
tarball="$cache_dir/vulkan-sdk-${SDK_VERSION}.0.tar.xz"

if [ -n "${MOE4ALL_SDK_TARBALL:-}" ]; then
  echo "==> Using local tarball override: $MOE4ALL_SDK_TARBALL"
  cp -f "$MOE4ALL_SDK_TARBALL" "$tarball"
fi
if [ -f "$tarball" ]; then
  echo "==> Verifying cached tarball (full archive listing)..."
  if tar -tJf "$tarball" >/dev/null 2>&1; then
    echo "    Cache OK."
  else
    echo "    Cached tarball failed integrity check — will re-download."
    mv -f "$tarball" "$tarball.bad"
  fi
fi
if [ ! -f "$tarball" ]; then
  echo "==> Downloading LunarG Vulkan SDK ${SDK_VERSION} (~314 MB, resumable)..."
  # .part + rename: a killed/corrupt transfer never masquerades as a complete file.
  curl -fSL --retry 3 --retry-delay 5 -C - -o "$tarball.part" "$SDK_TARBALL_URL"
  mv -f "$tarball.part" "$tarball"
  echo "==> Verifying archive integrity (full listing)..."
  if ! tar -tJf "$tarball" >/dev/null 2>&1; then
    echo "error: tarball failed integrity check (download likely corrupted)."
    echo "       Re-run the script — the download resumes — or delete the cache file:"
    echo "       rm $tarball"
    exit 1
  fi
fi

# 3) Extract glslc by EXACT member path (discovered, not globbed), plus the shared
#    libraries glslc links against (per ldd). Exact names => identical behavior
#    across GNU tar and bsdtar.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/extract"

members="$(tar -tJf "$tarball")"
glslc_member="$(printf '%s\n' "$members" | grep -E '(^|/)bin/glslc$' | head -1 || true)"
if [ -z "$glslc_member" ]; then
  echo "error: glslc not found in the SDK tarball. First lines of the archive:"
  printf '%s\n' "$members" | head -20
  echo "       (If the archive looks well-formed but has no bin/glslc, the SDK"
  echo "        layout may have changed — try --sdk-version with a known release,"
  echo "        or set MOE4ALL_SDK_TARBALL to a local copy.)"
  exit 1
fi
echo "==> Extracting $glslc_member"
extract_list=("$glslc_member")
glslc_extracted="$work/extract/$glslc_member"
tar -xJf "$tarball" -C "$work/extract" "$glslc_member"

# Shared libraries glslc links against: prefer the SDK's own copies (exact match,
# guaranteed ABI), fall back to system. Only extract ones present in the archive.
# (`|| true` on each pipeline: under `set -o pipefail` a grep with no match would
# otherwise abort the script silently.)
libnames="$(ldd "$glslc_extracted" 2>/dev/null | awk '{print $1}' | grep -E '^lib.*\.so' | sort -u || true)"
for libname in $libnames; do
  member="$(printf '%s\n' "$members" | grep -E "(^|/)$libname\$" | head -1 || true)"
  if [ -n "$member" ]; then
    extract_list+=("$member")
  fi
done
if [ "${#extract_list[@]}" -gt 1 ]; then
  echo "    + $(( ${#extract_list[@]} - 1 )) runtime libraries"
  tar -xJf "$tarball" -C "$work/extract" "${extract_list[@]:1}"
fi

# 4) Lay it down under $PREFIX with a self-contained wrapper (the wrapper sets
#    LD_LIBRARY_PATH, so no system config is touched).
install_lib="$PREFIX/lib/moe4all-glslc"
install_bin="$PREFIX/bin"
mkdir -p "$install_lib" "$install_bin"
cp -f "$work/extract/$glslc_member" "$install_lib/glslc"
chmod +x "$install_lib/glslc"
for m in "${extract_list[@]:1}"; do
  cp -Lf "$work/extract/$m" "$install_lib/" 2>/dev/null || true
done

wrapper="$install_bin/glslc"
cat > "$wrapper" <<EOF
#!/bin/sh
# Installed by MoE4All scripts/install-linux.sh — LunarG Vulkan SDK glslc ${SDK_VERSION}.
exec env LD_LIBRARY_PATH="$install_lib:\$LD_LIBRARY_PATH" "$install_lib/glslc" "\$@"
EOF
chmod +x "$wrapper"

echo "==> Verifying the installed glslc can compile GL_KHR_cooperative_matrix..."
if ! glslc_ok "$wrapper"; then
  echo "error: installed glslc failed the cooperative-matrix probe."
  echo "       Inspect: $wrapper --version   and   ldd $install_lib/glslc"
  exit 1
fi
echo "    OK: $wrapper"
"$wrapper" --version 2>/dev/null | head -1 || true

# 5) Make sure $PREFIX/bin is on PATH for future shells.
case ":$PATH:" in
  *":$install_bin:"*) echo "==> $install_bin is already on PATH.";;
  *)
    echo "==> Add this to your shell rc (~/.bashrc or ~/.zshrc) to put glslc on PATH:"
    echo "    export PATH=\"$install_bin:\$PATH\""
    if [ -f "$HOME/.bashrc" ]; then
      if ! grep -qF "$install_bin" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n# MoE4All glslc (installed by scripts/install-linux.sh)\nexport PATH="%s:$PATH"\n' "$install_bin" >> "$HOME/.bashrc"
        echo "    (appended to ~/.bashrc)"
      fi
    fi
  ;;
esac

echo
echo "==> Done. Now build MoE4All:"
echo "    cd <MoE4All checkout>"
echo "    export PATH=\"$install_bin:\$PATH\"   # or open a new shell"
echo "    cargo build --release --locked -p infr-cli"
echo "    ./target/release/infr devices"
echo
echo "    (Runtime deps for the Vulkan path: a Vulkan driver + libvulkan, e.g."
echo "     Ubuntu: sudo apt install libvulkan1 vulkan-tools  — only needed to RUN, not to build.)"
