#!/usr/bin/env bash
#
# push_webui_to_hf.sh — convenience wrapper around push_data_to_hf.py for the
# WebUI dataset. Picks sensible defaults from the standard UIPro layout but
# lets you override anything via environment variables.
#
# Quick test (no network):
#   bash utils/data_utils/push_webui_to_hf.sh --dry-run
#
# Real upload (set the repo first):
#   REPO_ID=your-username/uipro-webui bash utils/data_utils/push_webui_to_hf.sh
#
# Override the samples file (e.g. to push a specific run):
#   SAMPLES=data/WebUI_processed/WebUI_scale1000_149k.json \
#     bash utils/data_utils/push_webui_to_hf.sh --dry-run
#
# Other env vars: IMAGES_ROOT, PYTHON, TOKEN (or pre-set HF_TOKEN).
# Anything after the script name is forwarded verbatim to push_data_to_hf.py
# (e.g. --private, --skip-zip, --commit-message "...").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
IMAGES_ROOT="${IMAGES_ROOT:-$REPO_ROOT/data}"
REPO_ID="${REPO_ID:-your-username/uipro-webui}"

# Pick the most recently modified WebUI_scale*_<N>k.json under
# data/WebUI_processed/ unless the caller overrode it. Excludes the
# `_sample.json` and `_info.json` sibling files.
if [[ -z "${SAMPLES:-}" ]]; then
    SAMPLES=$(
        find "$REPO_ROOT/data/WebUI_processed" \
            -maxdepth 1 -type f -name 'WebUI_scale*k*.json' \
            ! -name '*_sample.json' ! -name '*_info.json' \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | cut -d' ' -f2-
    )
fi

if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
    echo "ERROR: could not locate a WebUI samples .json." >&2
    echo "  Looked under: $REPO_ROOT/data/WebUI_processed/" >&2
    echo "  Set SAMPLES=/path/to/your.json to override." >&2
    exit 1
fi
if [[ ! -d "$IMAGES_ROOT" ]]; then
    echo "ERROR: IMAGES_ROOT is not a directory: $IMAGES_ROOT" >&2
    exit 1
fi
if [[ "$REPO_ID" == "your-username/uipro-webui" ]]; then
    echo "WARN: REPO_ID still set to the placeholder 'your-username/uipro-webui'." >&2
    echo "      Set REPO_ID=<owner>/<repo> for a real upload (or pass --dry-run)." >&2
fi

echo "Samples     : $SAMPLES"
echo "Images root : $IMAGES_ROOT"
echo "Repo ID     : $REPO_ID"
echo "Python      : $PYTHON"
echo "Extra args  : $*"
echo

# Pass the optional --token only when TOKEN is set, so we don't shadow
# $HF_TOKEN / the cached login when the user hasn't supplied one.
TOKEN_ARG=()
if [[ -n "${TOKEN:-}" ]]; then
    TOKEN_ARG=(--token "$TOKEN")
fi

exec "$PYTHON" "$REPO_ROOT/utils/data_utils/push_data_to_hf.py" \
    --samples "$SAMPLES" \
    --images-root "$IMAGES_ROOT" \
    --repo-id "$REPO_ID" \
    "${TOKEN_ARG[@]}" \
    "$@"
