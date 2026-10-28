"""Write data/soy_family_folds.csv: leave-families-out splits for the SoyNAM soybean data.

In each of 10 replicates, 8 of the 38 SoyNAM families (line IDs DS11-<family><line>) are
held out at random (fold 1) and the other 30 are used for training (fold 2). The file has
the same layout as soy_CVFs.csv, so it can be passed to tabpfn_holdout.py and
comparison_holdout.py with --folds.

Usage: python data/make_family_folds.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

ids = pd.read_csv(DATA_DIR / "soy_pheno.csv", index_col=0).index
family = np.array([i[:7] for i in ids])  # e.g. DS11-02
families = np.unique(family)
rng = np.random.default_rng(2026)
folds = {}
for rep in range(1, 11):
    held = rng.choice(families, size=8, replace=False)
    folds[f"cv_{rep}"] = np.where(np.isin(family, held), 1, 2)
out = pd.DataFrame(folds, index=ids)
out.index.name = "ID"
out.to_csv(DATA_DIR / "soy_family_folds.csv")
print(f"{len(families)} families; held-out lines per replicate:",
      (out == 1).sum().tolist())
