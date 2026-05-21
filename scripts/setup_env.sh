#!/usr/bin/env bash
# UNIX/Linux setup automation script for virtual environments

set -e

echo "====================================================================="
echo "🛠️  NVIDIA NEMOTRON REASONING PIPELINE: UNIX ENVIRONMENT SETUP"
echo "====================================================================="
echo

# Resolve directory context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python Virtual Environment (.venv)..."
    python3 -m venv .venv
else
    echo "✨ Virtual environment (.venv) already exists."
fi

# Activate environment
echo "⚡ Activating Virtual Environment..."
source .venv/bin/activate

# Upgrade pip & install package requirements
echo "🚀 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo
echo "====================================================================="
echo "✅ ENVIRONMENT SETUP SUCCESSFUL!"
echo "====================================================================="
echo
echo "To activate this environment, execute:"
echo "  source .venv/bin/activate"
echo
echo "To run unit tests to verify the setup:"
echo "  pytest"
echo
echo "To run the pipeline in debug/prep mode:"
echo "  python scripts/run_pipeline.py --phase prep --limit 5"
echo "====================================================================="
