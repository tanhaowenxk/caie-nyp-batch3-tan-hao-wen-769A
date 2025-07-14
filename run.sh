#!/bin/bash
set -e

# Check user input
if [ $# -eq 0 ]; then
  echo "Please specify a model type: logreg, rf, xgb, or all"
  echo "Usage: ./run.sh [logreg|rf|xgb|all]"
  read -p "Press enter to exit"
  exit 1
fi

MODEL=$1

if [[ "$MODEL" == "all" ]]; then
  echo "Running all models (logreg, rf, xgb)..."
  for m in logreg rf xgb
  do
    echo "Running model: $m"
    python src/main.py "$m"
    echo "Finished: $m"
    echo ""
  done
elif [[ "$MODEL" == "logreg" || "$MODEL" == "rf" || "$MODEL" == "xgb" ]]; then
  echo "Running model: $MODEL"
  python src/main.py "$MODEL"
else
  echo "Invalid model type: $MODEL"
  echo "Please choose: logreg, rf, xgb, or all"
  read -p "Press enter to exit"
  exit 1
fi

# Keep terminal open
read -p "Press enter to exit"
