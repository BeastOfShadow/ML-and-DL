#!/usr/bin/env bash
# WEKA regression on the same protocol as the notebook:
# 80/20 holdout (seed 1) + Normalize to [0,1] (matches MinMaxScaler).
# Three models: linear regression via SGD, KNN regression, Decision Tree regression.
#
# Usage:
#   ./run_weka_regression.sh
#   WEKA_JAR=/path/to/weka.jar ./run_weka_regression.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARFF="$SCRIPT_DIR/california_housing.arff"
OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"

WEKA_JAR="${WEKA_JAR:-}"
if [[ -z "$WEKA_JAR" ]]; then
  for c in \
    /Applications/weka-*.app/Contents/app/weka.jar \
    "$HOME"/Applications/weka-*.app/Contents/app/weka.jar \
    /Applications/weka-*/weka.jar \
    "$HOME"/weka-*/weka.jar \
    /usr/share/java/weka.jar; do
    [[ -f "$c" ]] && WEKA_JAR="$c" && break
  done
fi
if [[ ! -f "$WEKA_JAR" ]]; then
  echo "weka.jar not found. Set the path:  WEKA_JAR=/path/to/weka.jar $0" >&2
  exit 1
fi
echo "weka.jar: $WEKA_JAR"
echo "dataset:  $ARFF"
echo

NORM="weka.filters.unsupervised.attribute.Normalize"   # scale to [0,1] = MinMaxScaler
FC="weka.classifiers.meta.FilteredClassifier"
COMMON=(-t "$ARFF" -split-percentage 80 -no-cv -s 1)

NAMES=(LinearRegression_SGD KNN DecisionTree)
CLF=(
  "weka.classifiers.functions.SGD -- -F 2"   # linear regression by SGD (squared loss)
  "weka.classifiers.lazy.IBk -- -K 5"        # KNN regression
  "weka.classifiers.trees.REPTree"           # regression tree (~ CART)
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  out="$OUT_DIR/reg_$name.txt"
  echo "=== $name ==="
  java -cp "$WEKA_JAR" "$FC" "${COMMON[@]}" -F "$NORM" -W ${CLF[$i]} > "$out" 2>&1
  grep "Correlation coefficient"  "$out" | head -1 || true
  grep "Mean absolute error"      "$out" | head -1 || true
  grep "Root mean squared error"  "$out" | head -1 || true
  echo "  full output: $out"
  echo
done

echo "Fill weka_results in the notebook ([R2, MAE, RMSE]):"
echo "  R2   = (Correlation coefficient)^2"
echo "  MAE  = Mean absolute error"
echo "  RMSE = Root mean squared error"
