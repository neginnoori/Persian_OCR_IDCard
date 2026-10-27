#!/bin/zsh
cd "${0:A:h}"
exec .venv/bin/jupyter lab --notebook-dir="$PWD" notebooks/Persian_Card_OCR_v2.ipynb
