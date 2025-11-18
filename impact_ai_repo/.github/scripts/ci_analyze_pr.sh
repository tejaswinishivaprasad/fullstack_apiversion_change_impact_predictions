#!/bin/bash
set -euo pipefail

echo "=== Starting AI Core CI analysis ==="

# Move into the ai-core/src folder so server.py sees datasets/curated correctly
cd impact_ai_repo/ai-core/src

echo "Working directory: $(pwd)"

echo "Installing Python requirements..."
pip install -r requirements.txt

# Optional but safe for demo: generate small curated dataset
echo "Running curated dataset generator..."
python3 curated_datasets.py --out datasets/curated --max 6 --seed 123

# Now run analysis using helper script
echo "Running PR impact analysis..."
python3 run_analysis_ci.py

if [ -f pr-impact-report.json ]; then
    echo "PR impact report generated successfully."
else
    echo "ERROR: pr-impact-report.json not created"
    exit 1
fi

echo "=== AI Core CI analysis done ==="
