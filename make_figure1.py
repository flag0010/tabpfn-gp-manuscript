"""Figure 1: TabPFN-3.5 minus the best comparison model, per species and trait.

Reads results/paired_best.csv (run compare_models.py first) and writes
manuscript/figures/figure1.pdf and figure1.png.

Usage: python make_figure1.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript" / "figures"

NAME = {"soy": "Soybean", "spruce": "White spruce", "switchgrass": "Switchgrass",
        "sorghum": "Sorghum", "maize": "Maize", "rice": "Rice"}
N_TRAIN = {"soy": 4011, "spruce": 1377, "switchgrass": 412, "sorghum": 361, "maize": 313,
           "rice": 262}
MODEL = {"GBLUP": "GBLUP", "RKHS": "RKHS", "GBLUP_thin": "GBLUP-thin"}
# TabPFN's input per species (see tabpfn_holdout.py and compare_models.main_run)
INPUT = {"soy": "raw markers", "spruce": "raw markers"}
HYBRID = "PCs + thinned markers"

TEXT, TEXT_2, GRID, AXIS = "#1a1a1a", "#5b5b57", "#e6e5e1", "#9a9993"
WIN, LOSS, TIE = "#2a78d6", "#d64545", "#b9b8b2"


def colour(row):
    return WIN if row.ci_low > 0 else LOSS if row.ci_high < 0 else TIE


def main():
    d = pd.read_csv(ROOT / "results" / "paired_best.csv")
    species = list(dict.fromkeys(d.species))

    # One row per test, with a header row before each species
    ys, y, heads = [], 0.0, []
    for sp in species:
        heads.append((sp, y))
        y += 1.0
        for _ in range((d.species == sp).sum()):
            ys.append(y)
            y += 1.0
        y += 0.4
    d["y"] = ys

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": TEXT})
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * y + 0.9))
    ax.axvline(0, color=AXIS, lw=1.1, zorder=1)
    for x in (-0.05, 0.05, 0.10):
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)

    for row in d.itertuples():
        c = colour(row)
        ax.plot([row.ci_low, row.ci_high], [row.y, row.y], color=c, lw=1.8, zorder=2,
                solid_capstyle="butt")
        ax.plot(row.diff, row.y, "o", ms=5.5, color=c, zorder=3)
        ax.text(-0.152, row.y, f"{row.trait}", ha="right", va="center", color=TEXT)
        ax.text(-0.149, row.y, f"vs {MODEL[row.model]}", ha="left", va="center",
                color=TEXT_2, fontsize=7.5)
        ax.text(0.128, row.y, f"{row.tabpfn_wins}/10", ha="right", va="center", color=TEXT_2)
    ax.set_xlim(-0.29, 0.13)
    ax.set_ylim(y + 0.5, -1.8)
    fig.tight_layout()  # final axes size, so the name widths below are measured correctly
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    for sp, yh in heads:
        name = ax.text(-0.29, yh, NAME[sp], ha="left", va="center", fontweight="bold",
                       color=TEXT)
        # Start the grey label just after the bold name, whatever its width
        x_end = to_data.transform(name.get_window_extent(renderer).p1)[0]
        ax.text(x_end + 0.006, yh,
                f"{N_TRAIN[sp]:,} training lines · {INPUT.get(sp, HYBRID)}",
                ha="left", va="center", color=TEXT_2, fontsize=8)
    ax.text(0.128, -1.1, "splits won", ha="right", va="center", color=TEXT_2, fontsize=8)

    ax.set_xticks([-0.05, 0, 0.05, 0.10], ["−0.05", "0", "+0.05", "+0.10"])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.08, 0.10)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=TEXT_2, length=3)
    ax.set_xlabel("Δ Pearson r (TabPFN-3.5 minus best comparison model)", color=TEXT, x=0.69)
    ax.text(-0.078, y - 0.1, "other model better", ha="left", va="center", color=TEXT_2,
            fontsize=7.5)
    ax.text(0.10, y - 0.1, "TabPFN better", ha="right", va="center", color=TEXT_2, fontsize=7.5)

    handles = [plt.Line2D([], [], marker="o", ls="-", color=c, ms=5, lw=1.8, label=lab)
               for c, lab in [(WIN, "95% interval above 0"), (LOSS, "below 0"),
                              (TIE, "includes 0")]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure1.{ext}", dpi=300, bbox_inches="tight")
    print(OUT / "figure1.pdf")


if __name__ == "__main__":
    main()
