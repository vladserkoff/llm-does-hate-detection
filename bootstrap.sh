#! /usr/bin/env bash

set -e

if ollama list | grep hate-detect; then
    echo "Model is already running"
else
    ollama create hate-detect -f config/Modelfile
    echo "Model is created"
fi

python -m flask run --host=0.0.0.0 --port=5000 --reload --debug
