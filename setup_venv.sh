#!/usr/bin/env bash
# Local development without Docker. Docker is the primary, supported path —
# this is a convenience for running the backend/tests in a plain virtualenv.
set -e

python -m venv genai-doc-assistant
# Activate:
#   macOS/Linux:  source genai-doc-assistant/bin/activate
#   Windows:      genai-doc-assistant\Scripts\activate
source genai-doc-assistant/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# NOTE for Windows users: requirements.txt pins `python-magic`, which needs the
# libmagic binary that ships with Linux/Docker but not Windows. On Windows run:
#   pip uninstall python-magic && pip install python-magic-bin

cp -n .env.example .env || true
echo "Done. Add your ANTHROPIC_API_KEY to .env, then run:"
echo "  uvicorn app.api.main:app --reload --port 8000"
echo "  streamlit run frontend/app.py"
