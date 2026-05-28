#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "ERROR: no .git directory found in $REPO_DIR" >&2
    exit 1
fi

cd "$REPO_DIR"

BRANCH="diffusion-ada"

TAGS_BEFORE=$(git tag --list)

git fetch origin "$BRANCH" --tags --force > /dev/null 2>&1

LOCAL=$(git rev-parse "refs/heads/$BRANCH")
REMOTE=$(git rev-parse "refs/remotes/origin/$BRANCH")

TAGS_AFTER=$(git tag --list)
NEW_TAGS=$(comm -13 <(echo "$TAGS_BEFORE" | sort) <(echo "$TAGS_AFTER" | sort))

HAS_NEW_COMMITS=false
[ "$LOCAL" != "$REMOTE" ] && HAS_NEW_COMMITS=true

if [ -z "$NEW_TAGS" ] && [ "$HAS_NEW_COMMITS" = false ]; then
    echo "No changes identified"
    exit 0
fi

if [ "$HAS_NEW_COMMITS" = true ]; then
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
        git pull origin "$BRANCH" > /dev/null 2>&1
    else
        git fetch origin "$BRANCH:$BRANCH" > /dev/null 2>&1
    fi
fi

if [ -z "$NEW_TAGS" ]; then
    exit 0
fi

while IFS= read -r tag; do
    if [[ "$tag" =~ ^experimento-[A-Za-z0-9]+$ ]]; then
        mkdir -p "$REPO_DIR/experiments/$tag"
        echo "Launching experiment for tag $tag..."
        sbatch "$REPO_DIR/scripts/ci/finetune_job.sbs" "$REPO_DIR/experiments/$tag"
    fi
done <<< "$NEW_TAGS"
