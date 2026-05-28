#!/bin/zsh

PROJECT_ROOT="/Users/yasardemirelli/cmd_code/Uni_UZH/SS26/Neuromorphic Intelligence/keyspotting"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

echo "Dynapse environment loaded"

jupyter lab
