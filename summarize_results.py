"""Compare every TabPFN run with the 18 methods in Azodi et al. 2019 Table S5.

Writes:
  results/all_runs.csv       one row per species/trait/TabPFN input: mean r over the 10
                             replicate holdouts, the best published method, TabPFN's rank
                             among all 19 methods, and (soy, spruce) our rrBLUP on the
                             same splits
  results/summary_table.csv  the same, for the main run per species only (MAIN_RUN)
  results/method_ranks.csv   TabPFN (main runs) added to Table S5 as a 19th method: each
                             method's median (and mean) rank over the 18 traits and its wins

The 95% intervals on TabPFN's mean r use the Nadeau-Bengio correction for repeated random
splits, since the 10 training sets overlap: variance = (1/10 + n_test/n_train) * SD^2.

Usage: python summarize_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SPECIES = ["soy", "spruce", "rice", "sorghum", "maize", "switchgrass"]

# Result-file suffix -> how the markers were given to TabPFN
INPUTS = {
    "TabPFN": "raw markers",
    "TabPFN_cover": "raw markers, all covered",
    "TabPFN_PCA": "PCA",
    "TabPFN_thin_cover": "thinned markers",
    "TabPFN_hybrid100_cover": "100 PCs + thinned markers",
    "TabPFN_all_cover": "all markers, past the 20k limit",
}

# Main run per species: all raw markers where they fit TabPFN-3.5 (spruce with enough ensemble
# members to cover all 6,930 markers), else 100 PCs + thinned markers
MAIN_RUN = {"soy": "TabPFN", "spruce": "TabPFN_cover", "rice": "TabPFN_hybrid100_cover",
            "sorghum": "TabPFN_hybrid100_cover", "maize": "TabPFN_hybrid100_cover",
            "switchgrass": "TabPFN_hybrid100_cover"}

# Phenotype column -> trait name used in Table S5
TRAIT_NAMES = {
    "HT": "height", "FT": "flowering time", "YLD": "yield", "MO": "grain moisture",
    "R8": "R8", "DBH": "DBH", "DE": "wood density", "ST": "standability",
    "AN": "anthesis date",
}


def summarize(species, run, s5):
    tab = pd.read_csv(RESULTS / f"{species}_{run}.csv")
    rr_path = RESULTS / f"{species}_rrBLUP.csv"
    rr = pd.read_csv(rr_path) if rr_path.exists() else None
    rows = []
    for code, g in tab.groupby("trait"):
        trait = TRAIT_NAMES[code]
        pub = s5[(s5.Species == species) & (s5.Trait == trait)]
        assert len(pub) == 18, (species, trait, len(pub))
        best = pub.loc[pub.r.idxmax()]
        r_tab = g.r.mean()
        k, n_test, n_train = len(g), g.n_test.iloc[0], g.n_train.iloc[0]
        half = stats.t.ppf(0.975, k - 1) * np.sqrt((1 / k + n_test / n_train) * g.r.var())
        row = dict(
            species=species, trait=trait, run=run, input=INPUTS[run],
            n_train=int(g.n_train.iloc[0]), n_test=int(g.n_test.iloc[0]),
            tabpfn_r=r_tab, tabpfn_sd=g.r.std(),
            tabpfn_ci95_low=r_tab - half, tabpfn_ci95_high=r_tab + half,
            best_published=best.Algorithm, best_published_r=best.r,
            published_rrBLUP_r=pub.loc[pub.Algorithm == "rrBLUP", "r"].item(),
            tabpfn_minus_best=r_tab - best.r,
            tabpfn_rank_of_19=int((pub.r > r_tab).sum()) + 1,
        )
        if rr is not None:
            m = g.merge(rr, on=["trait", "rep"], suffixes=("", "_rr"))
            row.update(our_rrBLUP_r=m.r_rr.mean(), tabpfn_minus_our_rrBLUP=(m.r - m.r_rr).mean(),
                       tabpfn_wins_matched=f"{int((m.r > m.r_rr).sum())}/{len(m)}")
        rows.append(row)
    return rows


def main():
    s5 = pd.read_csv(ROOT / "data" / "azodi2019_tableS5.csv")
    rows = []
    for species in SPECIES:
        for run in INPUTS:
            if (RESULTS / f"{species}_{run}.csv").exists():
                rows += summarize(species, run, s5)
    allruns = pd.DataFrame(rows)
    allruns.to_csv(RESULTS / "all_runs.csv", index=False)
    main_rows = allruns[allruns.run == allruns.species.map(MAIN_RUN)]
    main_rows.to_csv(RESULTS / "summary_table.csv", index=False)

    # Rank TabPFN among the 18 published methods on every trait
    ours = main_rows.rename(columns={"species": "Species", "trait": "Trait", "tabpfn_r": "r"})
    ours = ours[["Species", "Trait", "r"]].assign(Algorithm="TabPFN")
    allm = pd.concat([s5[["Species", "Trait", "Algorithm", "r"]], ours])
    allm["rank"] = allm.groupby(["Species", "Trait"]).r.rank(ascending=False, method="min")
    ranks = allm.groupby("Algorithm")["rank"].agg(
        mean_rank="mean", median_rank="median",
        wins=lambda x: int((x == 1).sum()), top3=lambda x: int((x <= 3).sum()),
    ).sort_values(["median_rank", "wins"], ascending=[True, False])
    ranks.to_csv(RESULTS / "method_ranks.csv")
    print(ranks.head(5).round(2).to_string(), "\n")

    # Wide view: mean r per input, next to the best published method
    wide = allruns.pivot_table(index=["species", "trait"], columns="input", values="tabpfn_r",
                               sort=False)
    best = allruns.drop_duplicates(["species", "trait"]).set_index(["species", "trait"])
    wide.insert(0, "best published", best.best_published_r)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
