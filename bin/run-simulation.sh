#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the CLI with substrate command
python3 -m simulation_toolkit.cli.main simulation "$@"