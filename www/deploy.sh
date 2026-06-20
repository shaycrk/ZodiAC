#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <filename>" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage

if [[ -z "${GMAPS_API_KEY:-}" ]]; then
  echo "Error: GMAPS_API_KEY is not set" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$1"
if [[ "$SRC" != /* ]]; then
  SRC="$SCRIPT_DIR/$SRC"
fi

if [[ ! -f "$SRC" ]]; then
  echo "Error: file not found: $SRC" >&2
  exit 1
fi

DEST_DIR="/var/www/html/ZodiAC"
BASENAME="$(basename "$SRC")"
DEST="$DEST_DIR/$BASENAME"
TMPFILE=""

cleanup() {
  [[ -n "$TMPFILE" && -f "$TMPFILE" ]] && rm -f "$TMPFILE"
}
trap cleanup EXIT

if [[ "$BASENAME" == *.html ]]; then
  TMPFILE="$(mktemp)"
  sed "s#{{GMAPS_API_KEY}}#${GMAPS_API_KEY}#g" "$SRC" > "$TMPFILE"
  DEPLOY_SRC="$TMPFILE"
else
  DEPLOY_SRC="$SRC"
fi

mkdir -p "$DEST_DIR"
cp "$DEPLOY_SRC" "$DEST"
chown root:root "$DEST"
chmod 644 "$DEST"
