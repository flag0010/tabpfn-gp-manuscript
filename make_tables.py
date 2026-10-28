"""Build the manuscript's tables from the results files.

Reads results/ (run summarize_results.py and compare_models.py first) and writes LaTeX table
bodies to manuscript/tables/, which manuscript/main.tex includes with \\input:

  ranks.tex      median rank among 19 methods, against Azodi et al. Table S5
  inputs.tex     mean accuracy by input for the four large species
  families.tex   soybean held-out-family accuracy
  tableS1.tex    per-test accuracy of every model

Each table is also written as CSV next to it, for checking.

Usage: python make_tables.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUT = ROOT / "manuscript" / "tables"
CODE = {"height": "HT", "R8": "R8", "yield": "YLD", "DBH": "DBH", "wood density": "DE",
        "flowering time": "FT", "grain moisture": "MO", "standability": "ST",
        "anthesis date": "AN"}
NAME = {"soy": "Soybean", "spruce": "White spruce", "switchgrass": "Switchgrass",
        "sorghum": "Sorghum", "maize": "Maize", "rice": "Rice"}
ORDER = ["soy", "spruce", "switchgrass", "sorghum", "maize", "rice"]
LARGE = ["rice", "sorghum", "switchgrass", "maize"]
MAIN = {"soy": "TabPFN", "spruce": "TabPFN_cover"}
MARKERS = {"rice": 57542, "sorghum": 56299, "switchgrass": 217150, "maize": 244781}


def mean_r(species, run, trait=None):
    """Mean r over replicates; a blank r (constant prediction) counts as 0."""
    f = R / f"{species}_{run}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if trait is not None:
        d = d[d.trait == CODE[trait]]
    return d.r.fillna(0).mean()


def n_undefined(species, run, trait=None):
    d = pd.read_csv(R / f"{species}_{run}.csv")
    if trait is not None:
        d = d[d.trait == CODE[trait]]
    return int(d.r.isna().sum())


def fmt(v, star=""):
    return "--" if v is None else f"{v:.3f}{star}"


def write(name, header, rows, colspec):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(OUT / f"{name}.csv", index=False)
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def table1():
    ranks = pd.read_csv(R / "method_ranks.csv").head(7)
    label = {"TabPFN": "TabPFN-3.5", "EN11": "EN11 (ensemble of 11 methods)",
             "EN5": "EN5 (ensemble of 5 methods)", "BRR": "Bayesian ridge regression"}
    rows = [[label.get(r.Algorithm, r.Algorithm), f"{r.median_rank:g}", r.wins, r.top3]
            for r in ranks.itertuples()]
    write("ranks", ["Method", "Median rank (of 19)", "Tests ranked first",
                           "Tests in top 3"], rows, "lrrr")


def table2():
    rows = []
    for sp in LARGE:
        enet = mean_r(sp, "ENet_hybrid")
        star = "*" if n_undefined(sp, "ENet_hybrid") else ""
        rows.append([NAME[sp], f"{MARKERS[sp]:,}",
                     fmt(mean_r(sp, "TabPFN_PCA")), fmt(mean_r(sp, "TabPFN_thin_cover")),
                     fmt(mean_r(sp, "TabPFN_hybrid100_cover")), fmt(mean_r(sp, "GBLUP")),
                     fmt(mean_r(sp, "GBLUP_thin")), fmt(enet, star)])
    write("inputs", ["Species", "Markers", "TabPFN: PCs", "TabPFN: thinned",
                            "TabPFN: hybrid", "GBLUP: all", "GBLUP: thinned", "ENet: hybrid"],
          rows, "lrrrrrrr")


def table3():
    rows = []
    for tr in ["height", "R8", "yield"]:
        star = "*" if n_undefined("soy", "ENet_family", tr) else ""
        rows.append([tr.capitalize() if tr != "R8" else "R8",
                     fmt(mean_r("soy", "GBLUP", tr)), fmt(mean_r("soy", "GBLUP_family", tr)),
                     fmt(mean_r("soy", "RKHS_family", tr)),
                     fmt(mean_r("soy", "ENet_family", tr), star),
                     fmt(mean_r("soy", "TabPFN_family", tr))])
    write("families", ["Trait", "GBLUP, random", "GBLUP, families",
                              "RKHS, families", "ENet, families", "TabPFN, families"],
          rows, "lrrrrr")


def table_s1():
    s5 = pd.read_csv(ROOT / "data" / "azodi2019_tableS5.csv")
    rows = []
    for sp in ORDER:
        large = sp in LARGE
        traits = sorted(s5[s5.Species == sp].Trait.unique())
        for tr in traits:
            tab = mean_r(sp, MAIN.get(sp, "TabPFN_hybrid100_cover"), tr)
            pub = s5[(s5.Species == sp) & (s5.Trait == tr)]
            best = pub.loc[pub.r.idxmax()]
            rank = 1 + int((pub.r > tab).sum())
            enet = mean_r(sp, "ENet_hybrid" if large else "ENet", tr)
            rows.append([NAME[sp], tr, fmt(tab), fmt(mean_r(sp, "GBLUP", tr)),
                         fmt(mean_r(sp, "RKHS", tr)),
                         fmt(mean_r(sp, "GBLUP_thin", tr)) if large else "--",
                         fmt(enet, r"$^\dagger$" if large else ""),
                         f"{best.r:.3f} ({best.Algorithm})", rank])
    write("tableS1", ["Species", "Trait", "TabPFN-3.5", "GBLUP", "RKHS", "GBLUP-thin",
                          "ENet", "Best published", "Rank of 19"], rows, "llrrrrrlr")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table_s1()
    for f in sorted(OUT.glob("*.csv")):
        print(f"\n{f.name}\n{pd.read_csv(f).to_string(index=False)}")
