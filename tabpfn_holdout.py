"""TabPFN regression on 10 replicate 80/20 holdouts (same splits as rrblup_holdout.R).

Replicate i uses column cv_i of the CVFs file; lines in fold TEST_FOLD are the test set.
No tuning is done on the training set.

Features (--features):
  raw  markers passed to TabPFN as-is (coded -1/0/1)
  pca  principal components of the markers, fit on the training lines only and with the
       test lines projected onto them. All components are kept (one per training line,
       at most), so species with more markers than TabPFN accepts still fit.
  thin raw markers, thinned to --max-features markers evenly spaced along the genome
       (markers sorted by the chromosome and position in their names). Neighboring
       markers are highly correlated, so thinning keeps most of the information while
       leaving every marker's own signal intact. Uses no trait data.
  hybrid the top --n-pcs principal components of all markers (fit on the training lines
       only) followed by raw markers thinned to fill the rest of --max-features. The
       components are given to every ensemble member (TabPFN's "constant_and_balanced"
       feature subsampling), so the genome-wide summary isn't diluted among the markers.

Ensemble size (--n-estimators): by default TabPFN-3.5 runs 8 ensemble members, each on a
random subset of at most 768 features, so with more than 8 x 768 = 6,144 features some
markers are never seen. "cover" uses the smallest count that covers every feature.

Usage: python tabpfn_holdout.py <species> [--features raw|pca|thin|hybrid] [--n-estimators N|cover]
                                [--reps N] [--traits HT ...] [--folds FILE --tag NAME]
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Features per ensemble member in TabPFN-3.5 (max_features_per_estimator in its warnings)
FEATURES_PER_ESTIMATOR = 768

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "results"


def thin_markers(names, max_features):
    """Column indices of max_features markers evenly spaced along the genome.

    Marker names look like <chromosome>_<position>[_...], e.g. Chr1_1031812, where the
    chromosome may itself contain underscores (maize contigs: B73V4_ctg10_925082) and a
    few positions are in mangled scientific notation (sorghum: X6_5.5e.07 = 55,000,000).
    """
    def key(name):
        parts = name.split("_")
        for i in range(1, len(parts)):
            try:
                return "_".join(parts[:i]), float(parts[i].replace("e.", "e+"))
            except ValueError:
                continue
        raise ValueError(f"no position in marker name {name!r}")

    keys = [key(n) for n in names]
    order = sorted(range(len(names)), key=keys.__getitem__)
    if len(order) <= max_features:
        return np.array(order)
    picks = np.linspace(0, len(order) - 1, max_features).round().astype(int)
    return np.array(order)[picks]


def pca_features(X, train, n_components=None):
    """Project all lines onto the principal components of the training lines.

    Keeps the top n_components, or every component with non-negligible variance if None.
    """
    mean = X[train].mean(axis=0)
    # Rows of vt are the components, sorted by variance explained
    _, s, vt = np.linalg.svd(X[train] - mean, full_matrices=False)
    keep = s > s[0] * 1e-6
    if n_components is not None:
        keep[n_components:] = False
    return ((X - mean) @ vt[keep].T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--traits", nargs="+", default=None)
    ap.add_argument("--test-fold", type=int, default=1)
    ap.add_argument("--features", choices=["raw", "pca", "thin", "hybrid"], default="raw")
    ap.add_argument("--max-features", type=int, default=20_000,
                    help="total features for --features thin/hybrid (TabPFN-3.5's feature limit)")
    ap.add_argument("--n-pcs", type=int, default=100,
                    help="principal components put first by --features hybrid")
    ap.add_argument("--ignore-limits", action="store_true",
                    help="allow more features than TabPFN-3.5's 20,000 limit (ignore_pretraining_limits)")
    ap.add_argument("--n-estimators", default=None,
                    help='ensemble size: an integer, or "cover" (default: TabPFN\'s own)')
    ap.add_argument("--tag", default=None, help="model name in the output (default from options)")
    ap.add_argument("--folds", default=None,
                    help="alternative fold file in the CVFs layout, e.g. data/soy_family_folds.csv")
    args = ap.parse_args()
    # Imported here so comparison_holdout.py can reuse the helpers without installing tabpfn
    from tabpfn import TabPFNRegressor

    args.tag = args.tag or {"raw": "TabPFN", "pca": "TabPFN_PCA", "thin": "TabPFN_thin",
                            "hybrid": f"TabPFN_hybrid{args.n_pcs}"}[args.features] + (
        "_all" if args.ignore_limits else "") + (
        "_cover" if args.n_estimators == "cover" else
        f"_n{args.n_estimators}" if args.n_estimators else "")

    X = pd.read_csv(DATA_DIR / f"{args.species}_geno.csv", index_col=0)
    Y = pd.read_csv(DATA_DIR / f"{args.species}_pheno.csv", index_col=0)
    cvs = pd.read_csv(args.folds or DATA_DIR / f"{args.species}_CVFs.csv", index_col=0)
    assert X.index.equals(Y.index) and X.index.equals(cvs.index)
    if args.features == "hybrid":
        X_all = X.to_numpy(dtype=np.float32)  # all markers, for the components
        X = X.iloc[:, thin_markers(list(X.columns), args.max_features - args.n_pcs)]
        print(f"Thinned to {X.shape[1]} markers + {args.n_pcs} components", flush=True)
    if args.features == "thin":
        X = X.iloc[:, thin_markers(list(X.columns), args.max_features)]
        print(f"Thinned to {X.shape[1]} markers", flush=True)
    X = X.to_numpy(dtype=np.float32)
    traits = args.traits or list(Y.columns)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.species}_{args.tag}.csv"
    # Resume: keep finished fits and skip them
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["trait"], r["rep"]) for r in rows}
    pcs = {}  # PCA depends only on the split, so share it across traits
    for trait in traits:
        y = Y[trait].to_numpy()
        for rep in range(1, args.reps + 1):
            if (trait, rep) in done:
                continue
            test = cvs[f"cv_{rep}"].to_numpy() == args.test_fold
            t0 = time.time()
            if args.features == "pca":
                if rep not in pcs:
                    pcs[rep] = pca_features(X, ~test)
                F = pcs[rep]
            elif args.features == "hybrid":
                if rep not in pcs:
                    pcs[rep] = np.hstack([pca_features(X_all, ~test, args.n_pcs), X])
                F = pcs[rep]
            else:
                F = X
            kw = {}
            n_const = args.n_pcs if args.features == "hybrid" else 0
            if n_const:
                kw["inference_config"] = {
                    "FEATURE_SUBSAMPLING_METHOD": "constant_and_balanced",
                    "FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT": n_const,
                }
            if args.n_estimators == "cover":
                # Every member sees the n_const components plus its share of the rest
                kw["n_estimators"] = max(8, math.ceil(
                    (F.shape[1] - n_const) / (FEATURES_PER_ESTIMATOR - n_const)))
            elif args.n_estimators:
                kw["n_estimators"] = int(args.n_estimators)
            if args.ignore_limits:
                kw["ignore_pretraining_limits"] = True
            model = TabPFNRegressor(random_state=rep, **kw)
            model.fit(F[~test], y[~test])
            yhat = model.predict(F[test])
            secs = time.time() - t0
            r = np.corrcoef(y[test], yhat)[0, 1]
            rows.append(dict(species=args.species, trait=trait, rep=rep, model=args.tag,
                             n_train=int((~test).sum()), n_test=int(test.sum()),
                             n_features=F.shape[1], n_estimators=kw.get("n_estimators", "default"),
                             r=r, seconds=secs))
            print(f"{args.species} {trait} rep{rep} r={r:.4f} ({secs:.1f}s)", flush=True)
            # Write after every fit so a crash keeps finished results
            pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
