#!/bin/bash

###############################################################################
# Complete Pipeline Runner for LSTM-Attention-SHAP Volatility Forecasting
# This script executes the full workflow from data generation to paper export
###############################################################################

echo "======================================================================"
echo "LSTM-ATTENTION-SHAP VOLATILITY FORECASTING - COMPLETE PIPELINE"
echo "======================================================================"
echo ""

# Set working directory
cd "$(dirname "$0")"

# Create necessary directories
echo "[1/8] Creating directories..."
mkdir -p data models figures paper tests

# Step 1: Generate synthetic data
echo ""
echo "[2/8] Generating synthetic dataset..."
python3 code/data_generator.py

# Step 2: Visualize architecture
echo ""
echo "[3/8] Creating architecture diagram..."
python3 code/visualize_architecture.py

# Step 3: Train model
echo ""
echo "[4/8] Training LSTM-Attention model..."
python3 code/train.py

# Step 4: Run full evaluation and explanation pipeline
echo ""
echo "[5/8] Running complete evaluation and explainability pipeline..."
python3 code/main_pipeline.py

# Step 5: Generate paper
echo ""
echo "[6/8] Generating publication-quality paper (Word + PDF)..."
python3 code/generate_paper.py

# Step 6: Create ZIP package
echo ""
echo "[7/8] Creating deliverable package..."
cd ..
zip -r volatility_project_package.zip volatility_project/ \
    -x "*/.*" -x "*/__pycache__/*" -x "*.pyc"
mv volatility_project_package.zip volatility_project/

echo ""
echo "[8/8] Running tests..."
python3 tests/test_smoke.py

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Generated outputs:"
echo "  - Data: data/synthetic_data.csv"
echo "  - Models: models/lstm_attention_model.h5"
echo "  - Figures: figures/*.png"
echo "  - Paper: paper/paper_final.docx, paper/paper_final.pdf"
echo "  - Package: volatility_project_package.zip"
echo ""
echo "To view results:"
echo "  - Open paper/paper_final.docx in Microsoft Word"
echo "  - View figures in figures/ directory"
echo "  - Run Jupyter notebook: jupyter notebook code/notebook.ipynb"
echo ""
