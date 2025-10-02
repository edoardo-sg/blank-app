#!/usr/bin/env bash
set -e
cd /workspaces/blank-app
source .venv/bin/activate
.venv/bin/python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8502}
