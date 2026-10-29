#!/bin/bash
# Convenient launcher for Used Car Valuation Web App
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if pgrep -f "streamlit run app.py" > /dev/null; then
    echo "App is already running! Open in your browser: http://localhost:8501"
else
    echo "Starting Used Car Valuation Engine on http://localhost:8501 ..."
    nohup "$SCRIPT_DIR/venv/bin/streamlit" run app.py --server.port 8501 --server.headless true > /dev/null 2>&1 &
    sleep 2
    echo "App started successfully in the background!"
    echo "Access your app anytime at: http://localhost:8501"
fi
