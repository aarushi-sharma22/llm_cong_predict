#!/usr/bin/env bash
# One-time git setup for this repo.
# Usage from the project root:  bash scripts/setup_git.sh
set -euo pipefail

REPO_URL="git@github.com:aarushi-sharma22/llm-cong-predict-replication.git"
# HTTPS alternative:
# REPO_URL="https://github.com/aarushi-sharma22/llm-cong-predict-replication.git"

git init
git add .
git commit -m "Initial scaffold: config, CV metric layer (ported + tested), docs"
git branch -M main
git remote add origin "$REPO_URL"

echo
echo "Committed locally and remote set to: $REPO_URL"
echo "If the GitHub repo is EMPTY, push with:"
echo "    git push -u origin main"
echo
echo "If GitHub auto-added a README/license (repo not empty), first run:"
echo "    git pull --rebase origin main   # then resolve, then:"
echo "    git push -u origin main"
