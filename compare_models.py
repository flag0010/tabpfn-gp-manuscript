"""Paired comparison of TabPFN-3.5 with the comparison models on identical splits.

For each species, trait and comparison model: mean accuracy of both, the mean
per-replicate difference (TabPFN minus the other model), a 95% interval from the corrected
resampled t-test of Nadeau & Bengio (2003) with test/train ratio 0.25, and the number of
replicates on which TabPFN was more accurate.

A blank r in a results file means the model predicted a constant (an elastic net that kept
no markers), so the correlation is undefined; it is counted as r = 0 and the number of such
fits is reported in `undefined_fits`.

Writes:
  results/paired_comparison.csv  every species x trait x comparison model, for the random
                                 splits and (soybean) the held-out-family splits
  results/paired_best.csv        per test, TabPFN vs the most accurate of GBLUP, RKHS and
                                 GBLUP-thin (the data behind Figure 1)

Usage: python compare_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parent / "results"
SPECIES = ["soy", "spruce", "switchgrass", "sorghum", "maize", "rice"]
TRAITS = {"HT": "height", "R8": "R8", "YLD": "yield", "DBH": "DBH", "DE": "wood density",
          "FT": "flowering time", "MO": "grain moisture", "ST": "standability",
          "AN": "anthesis date"}
MODELS = ["GBLUP", "RKHS", "ENet", "GBLUP_thin", "ENet_hybrid"]
FIGURE1_MODELS = ["GBLUP", "RKHS", "GBLUP_thin"]
TEST_TRAIN_RATIO = 0.25  # 80/20 splits


def main_run(species):
    """Main TabPFN run per species: every input column seen by at least one ensemble member.

    Soybean's 4,234 markers fit within the default 8 members; spruce's 6,930 need 10
    ("cover"); the four large species use 100 PCs + 19,900 thinned markers.
    """
    return {"soy": "TabPFN", "spruce": "TabPFN_cover"}.get(species, "TabPFN_hybrid100_cover")


def load(species, model):
    f = R / f"{species}_{model}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)[["trait", "rep", "r", "seconds"]]
    d["undefined"] = d.r.isna().astype(int)
    d["r"] = d.r.fillna(0.0)
    return d


def corrected_interval(d, ratio=TEST_TRAIN_RATIO):
    k = len(d)
    se = np.sqrt((1 / k + ratio) * d.var(ddof=1))
    t = stats.t.ppf(0.975, k - 1)
    return d.mean() - t * se, d.mean() + t * se, 2 * stats.t.sf(abs(d.mean() / se), k - 1)


def compare(species, tab_model, suffix=""):
    rows = []
    tab = load(species, tab_model)
    if tab is None:
        return rows
    for model in MODELS:
        other = load(species, model + suffix)
        if other is None:
            continue
        m = tab.merge(other, on=["trait", "rep"], suffixes=("_t", "_o"))
        for trait, g in m.groupby("trait"):
            if len(g) < 10:
                continue
            d = (g.r_t - g.r_o).to_numpy()
            lo, hi, p = corrected_interval(d)
            rows.append(dict(
                species=species, trait=TRAITS[trait], design="family" if suffix else "random",
                tabpfn_run=tab_model, model=model, n_reps=len(g),
                tabpfn_r=g.r_t.mean(), model_r=g.r_o.mean(), diff=d.mean(),
                ci_low=lo, ci_high=hi, p=p, tabpfn_wins=int((d > 0).sum()),
                tabpfn_seconds=g.seconds_t.mean(), model_seconds=g.seconds_o.mean(),
                undefined_fits=int(g.undefined_o.sum())))
    return rows


def main():
    rows = []
    for sp in SPECIES:
        rows += compare(sp, main_run(sp))
    rows += compare("soy", "TabPFN_family", "_family")
    df = pd.DataFrame(rows)
    df.to_csv(R / "paired_comparison.csv", index=False)

    fig = df[(df.design == "random") & df.model.isin(FIGURE1_MODELS)]
    best = fig.loc[fig.groupby(["species", "trait"]).model_r.idxmax()].copy()
    best["species"] = pd.Categorical(best.species, SPECIES, ordered=True)
    best = best.sort_values(["species", "trait"])
    best.to_csv(R / "paired_best.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["species", "trait", "model", "tabpfn_r", "model_r", "diff", "ci_low",
                "ci_high", "tabpfn_wins"]
        print("TabPFN vs the best of GBLUP, RKHS and GBLUP-thin (Figure 1):")
        print(best[cols].round(3).to_string(index=False))
        print(f"\nAhead on {(best['diff'] > 0).sum()} of {len(best)}; interval above zero on "
              f"{(best.ci_low > 0).sum()}, below zero on {(best.ci_high < 0).sum()}")
        fam = df[df.design == "family"]
        if len(fam):
            print("\nHeld-out soybean families:")
            print(fam[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
