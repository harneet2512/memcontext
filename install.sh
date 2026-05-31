#!/usr/bin/env bash
# MemContext one-click install.
#
#   ./install.sh [DB_PATH]
#
# Installs the package into an isolated environment, verifies the stdio MCP
# server launches, and prints ready-to-paste MCP client config for BOTH
# Claude Code and Codex. Default DB: ~/.memcontext/memcontext.db
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_PATH="${1:-$HOME/.memcontext/memcontext.db}"
VENV_DIR="$REPO_DIR/.venv-memcontext"

mkdir -p "$(dirname "$DB_PATH")"
echo "[memcontext] repo: $REPO_DIR"
echo "[memcontext] db:   $DB_PATH"

activate() {
  # Cross-platform venv activation (Unix vs Windows/Git-Bash layout).
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"
}

# Prefer uv, then pipx, then a plain venv. All yield an isolated env with the
# `memcontext` console script and a resolvable interpreter.
if command -v uv >/dev/null 2>&1; then
  echo "[memcontext] installing with uv"
  uv venv "$VENV_DIR"
  activate
  uv pip install -e "$REPO_DIR[mcp,embeddings]"
elif command -v pipx >/dev/null 2>&1; then
  echo "[memcontext] installing with pipx"
  pipx install --force "$REPO_DIR[mcp,embeddings]"
else
  echo "[memcontext] installing with venv + pip"
  python3 -m venv "$VENV_DIR"
  activate
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -e "$REPO_DIR[mcp,embeddings]"
fi

# Resolve the interpreter the config should point at.
if command -v python >/dev/null 2>&1; then PY="$(command -v python)"; else PY="$(command -v python3)"; fi

echo "[memcontext] verifying stdio server launch..."
"$PY" -c "import memcontext.mcp_server as m; assert hasattr(m, 'main'); print('[memcontext] module import OK')"
if "$PY" "$REPO_DIR/scripts/smoke/mcp_smoke.py" >/dev/null 2>&1; then
  echo "[memcontext] stdio smoke: PASS"
else
  echo "[memcontext] stdio smoke: FAILED (is the 'mcp' extra installed?)" >&2
  exit 1
fi

echo ""
echo "=================== MCP client config ==================="
"$PY" -m memcontext.cli mcp-config --client both --db "$DB_PATH"
echo "========================================================="
echo "[memcontext] done. Paste the block above into your client, then restart it."
