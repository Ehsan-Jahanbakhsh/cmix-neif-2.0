#!/usr/bin/env bash
# Prepare the upstream training package from the original enwik9 on a Linux host.
set -eu
if [ "$#" -ne 2 ]; then
  echo "usage: bash prepare_enwik9.sh /path/to/enwik9 /path/to/work" >&2
  exit 2
fi
input=$(realpath -e "$1")
mkdir -p "$2"
work=$(realpath "$2")
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
base=35d1957a5074db802fc134126afc77b76d931761
entry="$work/entry-$base"
if [ ! -d "$entry" ]; then
  git -C "$repo" worktree add --detach "$entry" "$base"
fi
if [ "$(git -C "$entry" rev-parse --show-toplevel)" != "$entry" ] ||
   [ "$(git -C "$entry" rev-parse HEAD)" != "$base" ] ||
   ! git -c core.filemode=false -C "$entry" diff --quiet HEAD --; then
  echo "expected an unmodified checkout of $base in $entry" >&2
  exit 1
fi
export ENTRY="$entry" ENWIK9="$input" WORK="$work"
export DICT="$entry/dictionary/english.dic" ORDER="$entry/src/readalike_prepr/data/new_article_order"
exec bash "$entry/training/tf/2_package/build_package.sh"
