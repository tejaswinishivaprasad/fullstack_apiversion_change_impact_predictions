#!/bin/bash
# Simple script to run server.py analysis for CI demo

echo "Running AI Core analysis..."

python3 aicore/server.py --ci-scan 2> /dev/null

if [ -f impact-report.json ]; then
  mv impact-report.json pr-impact-report.json
  echo "Report generated."
else:
  echo "No report generated."
fi
