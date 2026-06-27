#!/usr/bin/env bash
# Local development without Docker. Docker is the primary, supported path —
# this sets up a plain virtualenv to run the app and the tests.
set -e

python -m venv .venv
# Activate:
#   macOS/Linux:  source .venv/bin/activate
#   Windows:      .venv\Scripts\activate   (or call .venv\Scripts\python.exe directly)
source .venv/bin/activate

pip install --upgrade pip
# Backend runtime + test toolchain; requirements-frontend adds Streamlit so you
# can run the UI too. (CPU-only torch comes from PyPI on Windows; on Linux add
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
# first to avoid the large CUDA wheels.)
pip install -r requirements-dev.txt -r requirements-frontend.txt

# NOTE for Windows users: requirements pin `python-magic`, which needs the libmagic
# binary that ships with Linux/Docker but not Windows. On Windows run:
#   pip uninstall python-magic && pip install python-magic-bin

echo "Done. Set ANTHROPIC_API_KEY in your shell (no .env file needed), then run:"
echo "  PowerShell:  \$env:ANTHROPIC_API_KEY = 'sk-ant-...'"
echo "  bash:        export ANTHROPIC_API_KEY=sk-ant-..."
echo "  uvicorn app.api.main:app --reload --port 8000"
echo "  streamlit run frontend/app.py"
