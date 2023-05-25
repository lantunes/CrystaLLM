#!/bin/bash
set -euxo pipefail

TEMP_DIR=$(mktemp -d)
ln -s "$(pwd)/lib" $TEMP_DIR

torch-model-archiver --model-name cif_model_19 --version 1.0 --model-file ./nanoGPT/model.py --serialized-file ./out/cif_model_19/ckpt.pt --export-path ./out/model_store --handler ./nanoGPT/serve_handler.py --extra-files $TEMP_DIR,serve_config.json

rm -rf $TEMP_DIR