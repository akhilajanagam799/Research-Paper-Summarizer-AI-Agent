#!/bin/bash

# Research Paper Summarizer - Repository Setup Script
# This script creates the complete directory structure and initializes the repository

echo "🚀 Setting up Research Paper Summarizer repository..."

# Create main directory structure
echo "📁 Creating directory structure..."

mkdir -p data/{raw,processed}
mkdir -p src/{agent,finetune,evaluate,ui}
mkdir -p docs
mkdir -p logs
mkdir -p models
mkdir -p scripts

# Create __init__.py files for Python packages
echo "📝 Creating Python package files..."

touch src/__init__.py
touch src/agent/__init__.py
touch src/finetune/__init__.py
touch src/evaluate/__init__.py
touch src/ui/__init__.py

# Create .gitkeep files for empty directories
echo "📄 Creating .gitkeep files..."

touch models/.gitkeep
touch logs/.gitkeep
touch data/raw/.gitkeep

# Create sample data placeholders
echo "📋 Creating sample data files..."

# Sample PDF placeholder
cat > data/raw/README.md << EOF
# Sample Research Papers

To test the system, download sample papers:

\`\`\`bash
# LoRA paper (perfect for testing)
curl -o sample_paper.pdf "https://arxiv.org/pdf/2106.09685.pdf"

# Attention paper
curl -o attention_paper.pdf "https://arxiv.org/pdf/1706.03762.pdf"

# GPT-3 paper  
curl -o gpt3_paper.pdf "https://arxiv.org/pdf/2005.14165.pdf"
\`\`\`

Note: Ensure you have permission to use these papers for academic purposes.
EOF

# Make scripts executable
echo "🔧 Setting permissions..."

chmod +x run_demo.sh
chmod +x scripts/create_repo.sh

# Initialize git repository (optional)
if command -v git &> /dev/null; then
    echo "📊 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Research Paper Summarizer project structure"
    echo "✅ Git repository initialized"
else
    echo "ℹ️  Git not found, skipping repository initialization"
fi

# Create virtual environment (optional)
if command -v python3 &> /dev/null; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo "ℹ️  Activate with: source venv/bin/activate"
else
    echo "⚠️  Python3 not found, please install Python 3.8+"
fi

echo ""
echo "✅ Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Download sample paper: curl -o data/raw/sample_paper.pdf 'https://arxiv.org/pdf/2106.09685.pdf'"
echo "4. Run demo: ./run_demo.sh"
echo "5. Or start Streamlit: streamlit run src/ui/streamlit_app.py"
echo ""
echo "📖 See README.md for detailed instructions"
echo "🏗️  Check docs/architecture.md for technical details"