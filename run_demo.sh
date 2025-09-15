#!/bin/bash

# Research Paper Summarizer Demo Script
# This script demonstrates the full pipeline

echo "🚀 Research Paper Summarizer Demo"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=.

# Download sample paper if not exists
if [ ! -f "data/raw/sample_paper.pdf" ]; then
    echo "📄 Downloading sample research paper..."
    mkdir -p data/raw
    curl -o data/raw/sample_paper.pdf "https://arxiv.org/pdf/2106.09685.pdf"
fi

# Run CLI demo
echo "🤖 Running CLI agent demo..."
python src/agent/run_agent.py --input data/raw/sample_paper.pdf --task all

# Launch Streamlit app
echo "🌐 Launching Streamlit interface..."
echo "📍 Navigate to: http://localhost:8501"
streamlit run src/ui/streamlit_app.py

echo "✅ Demo complete! Check the outputs in logs/ directory"