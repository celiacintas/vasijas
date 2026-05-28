#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "ERROR: no .git directory found in $REPO_DIR" >&2
    exit 1
fi

cd "$REPO_DIR"

BRANCH="diffusion-ada"

git fetch origin "$BRANCH" --tags

LOCAL=$(git rev-parse "$BRANCH")
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0
fi

TAGS_BEFORE=$(git tag --list)

git pull origin "$BRANCH"

TAGS_AFTER=$(git tag --list)
NEW_TAGS=$(comm -13 <(echo "$TAGS_BEFORE" | sort) <(echo "$TAGS_AFTER" | sort))

if [ -z "$NEW_TAGS" ]; then
    exit 0
fi

while IFS= read -r tag; do
    if [[ "$tag" =~ ^experimento-[A-Za-z0-9]+$ ]]; then
        mkdir -p "$tag"
        echo "$tag"
    fi
done <<< "$NEW_TAGS"
