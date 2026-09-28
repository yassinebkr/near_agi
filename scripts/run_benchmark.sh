#!/bin/sh
set -eu
uv run lda benchmark --suite smoke "$@"

