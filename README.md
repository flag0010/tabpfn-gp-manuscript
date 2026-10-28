# The Bitter Lesson comes to genomic prediction

Code, data and results for the manuscript *The Bitter Lesson comes to genomic prediction: an
untuned tabular foundation model matches or beats specialized methods* (Lex Flagel,
University of Minnesota). The manuscript is in [`manuscript/`](manuscript/): LaTeX source (`main.tex`) and the
built PDF (`main.pdf`).

We give TabPFN-3.5, a general-purpose pretrained tabular model, the marker matrix directly,
with no genetics-specific modeling and no tuning, and benchmark it on the six-species,
18-trait genomic prediction dataset of Azodi et al. (2019). This repository builds on our
TabPFN-3.5 hackathon entry (`flag0010/tabular-genomic-prediction-tabpfn-hackathon-2026`) and adds
comparison models rerun on identical splits, a held-out-family validation for soybean, and the
manuscript.

## Main results

- **On identical splits**, TabPFN-3.5 was ahead of the best of GBLUP, RKHS and GBLUP-thin on
  12 of 18 traits and significantly behind on only one (spruce wood density), with significant
  gains on all three soybean traits (+0.017 to +0.041 in Pearson r), maize yield and rice
  flowering time. See `results/paired_best.csv` and Figure 1.
- **Holding out whole soybean families** removed TabPFN's soybean advantage: all models lost
  most of their accuracy and TabPFN performed on par with GBLUP and RKHS
  (`results/paired_comparison.csv`, rows with `design == "family"`).
- **Against the 18 published methods**, TabPFN-3.5 had the best median rank and ranked first
  on 11 of 18 traits (`results/method_ranks.csv`). Published means come from different random
  splits, so the manuscript treats this ranking as descriptive.

## Repository layout

| Path | What it is |
|---|---|
| `data/` | Azodi et al. (2019) data: a checksum-verified mirror (`mirror/`), Table S5 of that paper, and the soybean held-out-family folds (`soy_family_folds.csv`, written by `make_family_folds.py`). See [`data/README.md`](data/README.md). |
| `tabpfn_holdout.py` | TabPFN-3.5 fits on 10 replicate 80/20 holdouts (raw, PCA, thinned or hybrid input). |
| `comparison_holdout.py` | GBLUP, RKHS, GBLUP-thin and elastic-net fits on the same holdouts (CPU only). |
| `rrblup_holdout.R` | rrBLUP in R, used to check that `comparison_holdout.py`'s GBLUP reproduces `rrBLUP::kin.blup`. |
| `summarize_results.py` | Ranks TabPFN among the 18 published methods. |
| `compare_models.py` | Paired comparisons of TabPFN with each comparison model, with corrected resampled t intervals. |
| `make_tables.py`, `make_figure1.py` | Build the manuscript's tables (`manuscript/tables/`) and Figure 1 (`manuscript/figures/`). |
| `results/` | Per-replicate results for every model and input (one CSV per species and model), plus the summary tables. |
| `run_all.sh` | Runs everything (see below). |

Model names in `results/`: `TabPFN` (raw markers, default ensemble), `TabPFN_cover` (enough
ensemble members to see every column), `TabPFN_hybrid100_cover` (100 PCs + 19,900 thinned
markers), `TabPFN_PCA`, `TabPFN_thin_cover`, `TabPFN_all_cover` (all rice markers, past the
20,000-column limit), and the comparison models `GBLUP`, `RKHS`, `GBLUP_thin`, `ENet`,
`ENet_hybrid`, `rrBLUP`. A `_family` suffix marks the soybean held-out-family splits. A blank
`r` means the model predicted a constant (an elastic net that kept no markers); the analysis
counts it as r = 0.

## Reproducing the results

Python 3.10 or newer. Unpack and verify the data first:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/prepare_data.py          # unpacks data/mirror and checks it against Dryad
```

**Tables, figure and PDF from the saved results** (no fitting; a minute):

```
./run_all.sh outputs
```

This needs a LaTeX installation with `latexmk` for the PDF; the tables and figure are written
without it.

**Refitting everything:**

```
./run_all.sh comparisons   # CPU only; about 1-2 hours on a laptop, most of it elastic net
./run_all.sh tabpfn        # needs a GPU and a Prior Labs account; about 22-93 s per fit
./run_all.sh rrblup        # optional check in R (package rrBLUP)
./run_all.sh outputs
```

The fitting scripts skip fits whose results are already saved, so delete the relevant
`results/*.csv` files first to refit from scratch. TabPFN runs in the manuscript used the
integrated GPU of an Apple M4 Pro chip. The first TabPFN run opens a browser to log in to Prior
Labs and accept the TabPFN-3.5 license; without a browser, accept it at
https://ux.priorlabs.ai and set `TABPFN_TOKEN` to your API key.

## Use of generative AI

Claude (Anthropic) helped write and run the comparison-model code, verify citations, and
draft and revise the manuscript; ChatGPT (OpenAI) reviewed a draft. See the manuscript's
"Use of generative AI" statement.

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Data: Azodi et al. (2019), CC0, from Dryad
(doi:10.5061/dryad.xksn02vb9). The TabPFN-3.5 model weights are not included; Prior Labs
licenses them separately for non-commercial use, and commercial use requires a separate license
agreement with Prior Labs.

## Citation

If you use this work, please cite the manuscript (preprint link to be added) and
Azodi CB et al. (2019), Benchmarking parametric and machine learning models for genomic
prediction of complex traits, *G3* 9(11):3691–3702, doi:10.1534/g3.119.400498.
