#!/bin/bash
# Quick start script for BixBench Explorer with UV

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with UV..."
    uv venv --python /opt/homebrew/bin/python3
    source .venv/bin/activate
    echo "Installing dependencies..."
    uv pip install flask pandas huggingface_hub datasets nbformat nbconvert
else
    source .venv/bin/activate
fi

echo "Starting BixBench Explorer..."
echo "Visit http://localhost:5001 in your browser"
python app.py
