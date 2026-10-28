#!/usr/bin/env bash
# Reproduce every result, table and figure in the manuscript.
# Each fitting script saves after every fit and skips finished fits, so rerunning resumes.
#
# Usage: ./run_all.sh [tabpfn|comparisons|rrblup|outputs|all]   (default: outputs)
#   tabpfn       TabPFN-3.5 fits (needs a GPU and a Prior Labs account)
#   comparisons  GBLUP, RKHS and elastic net fits (CPU only)
#   rrblup       rrBLUP check in R on soybean and spruce
#   outputs      tables, Figure 1 and the PDF from the saved results (no fitting)
#   all          everything above
set -euo pipefail
cd "$(dirname "$0")"
what="${1:-outputs}"
LARGE="rice sorghum maize switchgrass"   # more markers than TabPFN-3.5's 20,000 columns
FAMILY="--folds data/soy_family_folds.csv"

if [[ "$what" == tabpfn || "$what" == all ]]; then
  # Raw markers where they fit (soybean 4,234; spruce 6,930, with enough members to cover all)
  python tabpfn_holdout.py soy --features raw
  python tabpfn_holdout.py spruce --features raw
  python tabpfn_holdout.py spruce --features raw --n-estimators cover
  # Large species, main run: 100 PCs given to every member + 19,900 thinned markers
  for sp in $LARGE; do
    python tabpfn_holdout.py "$sp" --features hybrid --n-pcs 100 --n-estimators cover
  done
  # Alternative inputs (Table 2 and the PCA comparison)
  for sp in soy spruce $LARGE; do python tabpfn_holdout.py "$sp" --features pca; done
  for sp in $LARGE; do python tabpfn_holdout.py "$sp" --features thin --n-estimators cover; done
  python tabpfn_holdout.py rice --features raw --ignore-limits --n-estimators cover
  # Soybean, held-out families
  python tabpfn_holdout.py soy --features raw $FAMILY --tag TabPFN_family
fi

if [[ "$what" == comparisons || "$what" == all ]]; then
  for sp in soy spruce $LARGE; do python comparison_holdout.py "$sp" --models GBLUP RKHS; done
  for sp in $LARGE; do python comparison_holdout.py "$sp" --models GBLUP_thin ENet_hybrid; done
  for sp in spruce soy; do python comparison_holdout.py "$sp" --models ENet; done
  python comparison_holdout.py soy --models GBLUP RKHS ENet $FAMILY --tag _family
fi

if [[ "$what" == rrblup || "$what" == all ]]; then
  for sp in soy spruce; do Rscript rrblup_holdout.R "$sp" 10 1; done
fi

if [[ "$what" == outputs || "$what" == all ]]; then
  python summarize_results.py > /dev/null
  python compare_models.py
  python make_tables.py > /dev/null
  python make_figure1.py
  if command -v latexmk > /dev/null; then
    (cd manuscript && latexmk -pdf -interaction=nonstopmode -quiet main.tex)
  fi
fi
