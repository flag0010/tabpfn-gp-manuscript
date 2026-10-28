"""Comparison models (CPU only) on the same 10 replicate 80/20 holdouts as tabpfn_holdout.py.

Replicate i holds out the lines in fold --test-fold of column cv_i of the CVFs file
(or of --folds, e.g. data/soy_family_folds.csv for the held-out-family design).
A fit that predicts a constant (an elastic net that keeps no markers) is saved with a
blank r; compare_models.py counts it as r = 0.

Models (--models):
  GBLUP       REML GBLUP with the relationship matrix built as in Azodi et al.'s rrBLUP
              script (K = ZZ'/mean(diag), Z = column-standardized markers, all lines).
              Equivalent to rrBLUP::kin.blup; checked against rrblup_holdout.R on soy/spruce.
  RKHS        Same REML machinery with a Gaussian kernel exp(-d2 / median(d2)), d2 the
              mean squared distance between standardized marker profiles.
  GBLUP_thin  GBLUP with K built from the --max-features - --n-pcs markers that
              tabpfn_holdout.py --features hybrid keeps (the thinned subset).
  ENet        Elastic net (l1_ratio 0.5), penalty chosen by 5-fold CV inside the training lines.
              (Named ENet to avoid confusion with Azodi et al.'s EN5/EN11 ensembles.)
  ENet_hybrid Elastic net on TabPFN's hybrid input: top --n-pcs PCs (training lines only)
              + the thinned markers.

Usage: python comparison_holdout.py <species> --models GBLUP RKHS ... [--reps 10]
       python comparison_holdout.py soy --models GBLUP --folds data/soy_family_folds.csv --tag _family
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import ElasticNetCV

from tabpfn_holdout import pca_features, thin_markers

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / "data" / "npy"


def load(species):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{species}_geno.npy"
    names = CACHE / f"{species}_markers.txt"
    if not f.exists():
        # Parse line by line: pandas needs >6 GB for the widest files (maize, switchgrass,
        # >200k columns), more than a small machine has; this peaks at a few hundred MB.
        with open(DATA_DIR / f"{species}_geno.csv") as fh:
            header = fh.readline().rstrip("\n").split(",")
            ids, rows = [], []
            for line in fh:
                key, rest = line.rstrip("\n").split(",", 1)
                ids.append(key.strip('"'))
                # float32, not int8: rice has imputed, non-integer dosages
                rows.append(np.array(rest.split(","), dtype=np.float32))
        np.save(f, np.vstack(rows))
        names.write_text("\n".join(h.strip('"') for h in header[1:]))
        (CACHE / f"{species}_ids.txt").write_text("\n".join(ids))
    X = np.load(f)
    markers = names.read_text().split("\n")
    ids = (CACHE / f"{species}_ids.txt").read_text().split("\n")
    Y = pd.read_csv(DATA_DIR / f"{species}_pheno.csv", index_col=0)
    cvs = pd.read_csv(DATA_DIR / f"{species}_CVFs.csv", index_col=0)
    assert list(map(str, Y.index)) == ids and list(map(str, cvs.index)) == ids
    return X, markers, Y, cvs


def standardize(X):
    """Column-standardize (float32), dropping monomorphic markers."""
    sd = X.std(axis=0, ddof=1, dtype=np.float64)
    keep = sd > 0
    Z = X[:, keep]
    return ((Z - Z.mean(axis=0, dtype=np.float64)) / sd[keep]).astype(np.float32)


def grm(X):
    Z = standardize(X)
    K = (Z @ Z.T).astype(np.float64)
    return K / np.mean(np.diag(K))


def gaussian_kernel(X):
    Z = standardize(X)
    sq = np.sum(Z.astype(np.float64) ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (Z @ Z.T).astype(np.float64), 0) / Z.shape[1]
    h = np.median(d2[np.triu_indices_from(d2, 1)])
    return np.exp(-d2 / h)


def reml_blup(K, y, train, test):
    """REML fit of y = mu + g + e, g ~ N(0, Vg K), on training lines; BLUP of g for test."""
    s, U = np.linalg.eigh(K[np.ix_(train, train)])
    s = np.maximum(s, 0)
    ys, xs = U.T @ y[train], U.T @ np.ones(train.sum())
    n = train.sum()

    def fit(log_delta):
        d = s + np.exp(log_delta)
        xdx = np.sum(xs ** 2 / d)
        beta = np.sum(xs * ys / d) / xdx
        r = ys - xs * beta
        sigma2 = np.sum(r ** 2 / d) / (n - 1)
        nll = 0.5 * ((n - 1) * np.log(sigma2) + np.sum(np.log(d)) + np.log(xdx))
        return nll, beta, r / d

    grid = np.linspace(-10, 10, 81)
    best = grid[np.argmin([fit(g)[0] for g in grid])]
    opt = minimize_scalar(lambda g: fit(g)[0], bounds=(best - 0.25, best + 0.25), method="bounded")
    _, beta, w = fit(opt.x)
    g_test = K[np.ix_(test, train)] @ (U @ w)
    return beta + g_test


def elastic_net(F, y, train, test, seed):
    mu = F[train].mean(axis=0)
    m = ElasticNetCV(l1_ratio=0.5, alphas=30, cv=5, n_jobs=-1, random_state=seed,
                     selection="random", max_iter=2000, tol=1e-3)
    m.fit(F[train] - mu, y[train])
    return m.predict(F[test] - mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species")
    ap.add_argument("--models", nargs="+", default=["GBLUP"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--test-fold", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=20_000)
    ap.add_argument("--n-pcs", type=int, default=100)
    ap.add_argument("--folds", default=None,
                    help="alternative fold file (same layout as the CVFs file)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()

    X, markers, Y, cvs = load(args.species)
    if args.folds:
        cvs = pd.read_csv(args.folds, index_col=0)
    thin = thin_markers(markers, args.max_features - args.n_pcs)
    kernels = {}
    for model in args.models:
        out_path = OUT_DIR / f"{args.species}_{model}{args.tag}.csv"
        rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
        done = {(r["trait"], r["rep"]) for r in rows}
        hyb = {}
        for trait in Y.columns:
            y = Y[trait].to_numpy(dtype=np.float64)
            for rep in range(1, args.reps + 1):
                if (trait, rep) in done:
                    continue
                test = cvs[f"cv_{rep}"].to_numpy() == args.test_fold
                train = ~test
                t0 = time.time()
                if model in ("GBLUP", "RKHS", "GBLUP_thin"):
                    if model not in kernels:
                        kernels[model] = {"GBLUP": lambda: grm(X),
                                          "RKHS": lambda: gaussian_kernel(X),
                                          "GBLUP_thin": lambda: grm(X[:, thin])}[model]()
                    yhat = reml_blup(kernels[model], y, train, test)
                elif model == "ENet":
                    yhat = elastic_net(X, y, train, test, rep)
                elif model == "ENet_hybrid":
                    if rep not in hyb:
                        hyb[rep] = np.hstack([pca_features(X, train, args.n_pcs), X[:, thin]])
                    yhat = elastic_net(hyb[rep], y, train, test, rep)
                else:
                    raise ValueError(model)
                secs = time.time() - t0
                r = np.corrcoef(y[test], yhat)[0, 1]
                rows.append(dict(species=args.species, trait=trait, rep=rep, model=model + args.tag,
                                 n_train=int(train.sum()), n_test=int(test.sum()), r=r, seconds=secs))
                print(f"{args.species} {model} {trait} rep{rep} r={r:.4f} ({secs:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
