#!/usr/bin/env bash
# Confronto WEKA dei 5 classificatori sullo stesso protocollo del notebook:
# holdout 80/20 (seed 1) + Standardize (equivalente a StandardScaler di sklearn).
#
# Uso:
#   ./run_weka.sh                         # prova a trovare weka.jar in posizioni comuni
#   WEKA_JAR=/path/to/weka.jar ./run_weka.sh   # path esplicito
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARFF="$SCRIPT_DIR/dataset_esame.arff"
OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"

# Path a weka.jar: override con la variabile WEKA_JAR, altrimenti prova posizioni comuni.
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
  echo "weka.jar non trovato. Imposta il path:  WEKA_JAR=/path/to/weka.jar $0" >&2
  exit 1
fi
echo "weka.jar: $WEKA_JAR"
echo "dataset:  $ARFF"
echo

STD="weka.filters.unsupervised.attribute.Standardize"
FC="weka.classifiers.meta.FilteredClassifier"
COMMON=(-t "$ARFF" -split-percentage 80 -no-cv -s 1)

NAMES=(LogisticRegression NaiveBayes KNN DecisionTree MLP)
CLF=(
  "weka.classifiers.functions.Logistic"
  "weka.classifiers.bayes.NaiveBayes"
  "weka.classifiers.lazy.IBk -- -K 5"
  "weka.classifiers.trees.J48"
  "weka.classifiers.functions.MultilayerPerceptron"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  out="$OUT_DIR/$name.txt"
  echo "=== $name ==="
  # -W <classifier> -- <opts del classificatore>  (es. IBk -- -K 5)
  java -cp "$WEKA_JAR" "$FC" "${COMMON[@]}" -F "$STD" -W ${CLF[$i]} > "$out" 2>&1
  grep "Correctly Classified Instances" "$out" | head -1 || true
  grep "Weighted Avg" "$out" | head -1 || true
  echo "  output completo: $out"
  echo
done

echo "Fatto. Per riempire weka_results nel notebook:"
echo "  Accuracy = 'Correctly Classified Instances' %  / 100"
echo "  F1       = colonna F-Measure   (riga 'Weighted Avg')"
echo "  ROC_AUC  = colonna ROC Area    (riga 'Weighted Avg')"
