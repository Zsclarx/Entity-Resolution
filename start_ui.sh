#!/bin/bash

echo "🚀 Starting Entity Resolution Pipeline UI..."
echo "=============================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  Activating virtual environment..."
    source pipeline_env/bin/activate
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import streamlit, plotly" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All UI dependencies installed"
else
    echo "📦 Installing UI dependencies..."
    pip install -r requirements_ui.txt
fi

echo ""
echo "🎯 Running Entity Resolution Pipeline UI"
echo "📱 Open your browser to: http://localhost:8502"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start Streamlit
streamlit run app_pipeline.py --server.port 8502 