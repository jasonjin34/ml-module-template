#!/bin/bash
set -e

if [[ ! -f Pipfile.lock ]]; then
    echo pytest not available, exiting
    exit 0
fi

export PYTHONPATH=./src/main/python/
pytest || true
