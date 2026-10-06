#!/usr/bin/env bash
# Project entry point. The virtualenv lives on a NATIVE filesystem on purpose:
# importing transformers from the DrvFs mount (/mnt/d, 9p) takes ~4.6 s and
# loading the tokenizer from an HF cache there takes ~23 s, per process.
# Override with SCP_PYTHON=... if you relocate it.
exec "${SCP_PYTHON:-$HOME/.venvs/scp-gpu/bin/python}" "$@"
