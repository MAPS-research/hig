#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HIG - model weight downloader
#
#   ./scripts/download_models.sh                 # default set: sdxl flux llm metrics
#   ./scripts/download_models.sh --only sdxl     # comma-separated group list
#   ./scripts/download_models.sh --check         # report what is already cached
#
# Groups
#   sdxl     SDXL base (fp16 variant) + the fp16-fix VAE          ~7.2G
#   flux     FLUX.1-dev, diffusers layout                        ~33G   [gated]
#   llm      Llama-3.1-8B *base* for soft-prompt embedding        ~16G
#
# meta-llama/Llama-3.1-8B is gated and its access form auto-rejects a lot of
# applicants, so 'llm' pulls the NousResearch mirror by default: same weights,
# same config, no gate. Override with LLAMA_REPO=meta-llama/Llama-3.1-8B if you
# did get access.
#   metrics  CLIP ViT-L/14 + LAION aesthetic predictor            ~1.7G
#
# Weights land in the HuggingFace cache ($HF_HOME/hub) so `from_pretrained`
# resolves them by repo id. The aesthetic predictor is a bare .pth on GitHub
# and goes to assets/ instead.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${HIG_ASSET_DIR:-$REPO_ROOT/assets}"
AESTHETIC_URL="https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"

# On the compute nodes an NYU site profile re-exports HF_HOME *after* ~/.bashrc
# has run, pointing it at a stale cache. Source the user rc back on top so the
# script lands weights in the same place interactive work does.
if [[ -f "$HOME/.bashrc" ]]; then
  set +eu
  # shellcheck source=/dev/null
  source "$HOME/.bashrc" >/dev/null 2>&1
  set -eu
fi

: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME

# a 50G download onto a small $HOME partition is the classic way to lose an
# afternoon; say something before it happens.
if [[ "$HF_HOME" == "$HOME"/* ]]; then
  echo "warning: HF_HOME=$HF_HOME is under \$HOME." >&2
  echo "         point it at scratch first, e.g.  export HF_HOME=/scratch/\$USER/cache/hf" >&2
  echo >&2
fi

LLAMA_REPO="${LLAMA_REPO:-NousResearch/Meta-Llama-3.1-8B}"

DEFAULT_SETS="sdxl flux llm metrics"
ALL_SETS="sdxl flux llm metrics"

# --- repo table: group, repo id, approx size, gated -------------------------
REPOS=(
  "sdxl    stabilityai/stable-diffusion-xl-base-1.0  6.9G  public"
  "sdxl    madebyollin/sdxl-vae-fp16-fix             0.4G  public"
  "flux    black-forest-labs/FLUX.1-dev              33G   gated"
  "llm     $LLAMA_REPO 16G   public"
  "metrics openai/clip-vit-large-patch14             1.7G  public"
)

# --- cli --------------------------------------------------------------------
SETS="$DEFAULT_SETS"
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) SETS="${2//,/ }"; shift 2 ;;
    --check) CHECK_ONLY=1; SETS="$ALL_SETS"; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

wants () { [[ " $SETS " == *" $1 "* ]]; }

cache_dir_for () { echo "$HF_HOME/hub/models--${1//\//--}"; }

# --- check mode -------------------------------------------------------------
if [[ $CHECK_ONLY -eq 1 ]]; then
  printf 'HF_HOME = %s\n\n' "$HF_HOME"
  printf '%-10s %-42s %-8s %s\n' GROUP REPO SIZE STATUS
  for row in "${REPOS[@]}"; do
    read -r grp repo size gate <<<"$row"
    d="$(cache_dir_for "$repo")"
    if [[ -d "$d" ]]; then
      status="cached ($(du -sh "$d" 2>/dev/null | cut -f1))"
    else
      status="MISSING${gate:+ [$gate]}"
      [[ "$gate" == public ]] && status="MISSING"
    fi
    printf '%-10s %-42s %-8s %s\n' "$grp" "$repo" "$size" "$status"
  done
  if [[ -f "$ASSET_DIR/sa_0_4_vit_l_14_linear.pth" ]]; then
    printf '%-10s %-42s %-8s %s\n' metrics "LAION aesthetic predictor" 3K cached
  else
    printf '%-10s %-42s %-8s %s\n' metrics "LAION aesthetic predictor" 3K MISSING
  fi
  exit 0
fi

# --- prerequisites ----------------------------------------------------------
if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "error: neither 'hf' nor 'huggingface-cli' found." >&2
  echo "       pip install -U huggingface_hub    (or: uv tool install huggingface_hub)" >&2
  exit 1
fi

fetch () {  # fetch <repo id> [hf download flags...]
  local repo="$1"; shift
  echo
  echo "=============================================================="
  echo ">>> $repo"
  echo "=============================================================="
  "${HF[@]}" "$repo" "$@"
}

echo "HF_HOME   = $HF_HOME"
echo "assets    = $ASSET_DIR"
echo "groups    = $SETS"
echo
df -h "$HF_HOME" 2>/dev/null || df -h "$(dirname "$HF_HOME")"
echo

# gated repos need an auth token; fail early rather than 40G in.
if wants flux; then
  if ! hf auth whoami >/dev/null 2>&1 && ! huggingface-cli whoami >/dev/null 2>&1; then
    cat >&2 <<'MSG'
error: FLUX.1-dev is a gated repo and no HF token was found.

  1. accept the license while logged in:
       https://huggingface.co/black-forest-labs/FLUX.1-dev
  2. log in:  hf auth login          (or export HF_TOKEN=hf_...)

  to skip it for now:  ./scripts/download_models.sh --only sdxl,llm,metrics
MSG
    exit 1
  fi
fi

# --- sdxl -------------------------------------------------------------------
if wants sdxl; then
  # fp16 variant only: skips the fp32 duplicates and the single-file
  # sd_xl_base_1.0.safetensors checkpoints (~17G of redundancy).
  # load with:  StableDiffusionXLPipeline.from_pretrained(..., variant="fp16")
  fetch stabilityai/stable-diffusion-xl-base-1.0 \
    --include "*.json" "*.txt" "*.fp16.safetensors"

  fetch madebyollin/sdxl-vae-fp16-fix \
    --include "*.json" "diffusion_pytorch_model.safetensors"
fi

# --- flux -------------------------------------------------------------------
if wants flux; then
  # keep the diffusers folder layout; drop the single-file ComfyUI checkpoints.
  fetch black-forest-labs/FLUX.1-dev \
    --exclude "flux1-dev.safetensors" "ae.safetensors" "*.gguf" "*.jpg" "*.png"
fi

# --- llm --------------------------------------------------------------------
if wants llm; then
  # base model, NOT -Instruct: the soft prompt is tuned for raw continuation
  # (eos 128001 = <|end_of_text|>, which the training objective relies on).
  # 'original/' holds a 16G consolidated .pth duplicate of the safetensors.
  fetch "$LLAMA_REPO" \
    --include "*.json" "*.txt" "*.safetensors" \
    --exclude "original/*"
fi

# --- metrics ----------------------------------------------------------------
if wants metrics; then
  fetch openai/clip-vit-large-patch14 \
    --include "*.json" "*.txt" "model.safetensors"

  mkdir -p "$ASSET_DIR"
  dst="$ASSET_DIR/sa_0_4_vit_l_14_linear.pth"
  if [[ -f "$dst" ]]; then
    echo ">>> aesthetic predictor already at $dst"
  else
    echo ">>> LAION aesthetic predictor -> $dst"
    curl -fL --retry 3 -o "$dst" "$AESTHETIC_URL"
  fi
fi

echo
echo "done. verify with:  ./scripts/download_models.sh --check"
