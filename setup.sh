#!/usr/bin/env bash
# One-shot environment setup for the memory-safe local-model benchmark.
#
#   bash setup.sh                # install deps + download models
#   bash setup.sh --no-models    # install deps only
#
# After it finishes:
#   source models/model_env.sh
#   python -m benchmark.run --pilot 12
#
# HF token: put HF_TOKEN=hf_xxx in .env (repo root or parent) before running --
# NV-Embed-v2 is gated, so downloads fail without an accepted license + token.

set -euo pipefail
cd "$(dirname "$0")"

DOWNLOAD_MODELS=1
[ "${1:-}" = "--no-models" ] && DOWNLOAD_MODELS=0

echo "== [1/3] base + benchmark Python deps =="
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-bench.txt

echo "== [2/3] llama-cpp-python (CUDA if available) =="
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    # Try a prebuilt CUDA wheel first; fall back to building from source with CUDA.
    python -m pip install -q llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
      || CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install -q --no-cache-dir --force-reinstall llama-cpp-python
else
    echo "   no CUDA detected -> CPU llama-cpp-python (inference will be slow)"
    python -m pip install -q llama-cpp-python
fi

echo "== [3/3] models =="
if [ "$DOWNLOAD_MODELS" = "1" ]; then
    python -m benchmark.download_models
else
    echo "   skipped (--no-models). Run later: python -m benchmark.download_models"
fi

echo
echo "Setup complete. Next:"
echo "  source models/model_env.sh"
echo "  python -m benchmark.run --pilot 12"
