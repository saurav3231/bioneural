#!/usr/bin/env bash
# BioNeural — one-click GitHub push (Linux/macOS)
#   ./push.sh [repo-name] [private|public]
set -euo pipefail

REPO_NAME="${1:-$(basename "$PWD")}"
VISIBILITY="${2:-private}"

command -v gh >/dev/null || { echo "ERROR: gh not found. Install from https://cli.github.com/" >&2; exit 1; }
[ "$VISIBILITY" = "private" ] || [ "$VISIBILITY" = "public" ] || { echo "visibility must be private|public" >&2; exit 1; }

git config user.name >/dev/null 2>&1 || { echo "ERROR: set git user.name / user.email first" >&2; exit 1; }
git config user.email >/dev/null 2>&1 || { echo "ERROR: set git user.name / user.email first" >&2; exit 1; }

[ -d .git ] || git init -b main
git add -A
git diff --cached --quiet || git commit -m "BioNeural v0.1 — event-driven ternary neural organism + benchmark harness"

if gh repo view "$REPO_NAME" --json nameWithOwner -q .nameWithOwner >/dev/null 2>&1; then
    git remote add origin "https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)" 2>/dev/null || true
    git push -u origin main
else
    gh repo create "$REPO_NAME" --"$VISIBILITY" --source . --remote origin --push
fi

echo "SUCCESS. Repo: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)"