# Data

All genotype, phenotype and cross-validation fold files come from Azodi et al. 2019,
deposited on Dryad under a CC0 (public domain) license:

> Azodi CB, Bolger E, McCarren A, Roantree M, de los Campos G, Shiu S-H (2019).
> Benchmarking parametric and machine learning models for genomic prediction of complex traits.
> G3 9(11):3691. https://doi.org/10.1534/g3.119.400498
> Data: https://doi.org/10.5061/dryad.xksn02vb9

There are two ways to get the data. Both end with the same verification step.

## Option 1: use the mirror in this repo (recommended)

We mirrored the 19 files here, gzip-compressed, in `data/mirror/` (45 MB in total). To unpack and verify them:

```
python data/prepare_data.py
```

This writes the 19 files into `data/` and checks each one's SHA-256 checksum against
the checksum Dryad publishes for it. 

## Option 2: download from Dryad by hand

1. Files at https://doi.org/10.5061/dryad.xksn02vb9
2. Download and put the files directly in `data/`.
3. Run `python data/prepare_data.py`. It finds the files already in place and verifies them.

## Expected checksums

`SHA256SUMS` lists the SHA-256 checksum Dryad publishes for each uncompressed file:


| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `maize_CVFs.csv` | 80942 | `0db49ecf31dd5db9d942bdc1baea9ae190bdfc17806a73869885fc46be02cf91` |
| `maize_geno.csv` | 217174436 | `a8162e3e93714eb1a5020f0187ff7f792163050c8f1eb63fd5ce9541edad6eeb` |
| `maize_pheno.csv` | 23462 | `4d8fee91731c0b035781283ee36ff91b8adbc58bf5edd8a408f253b2a25689f1` |
| `README.txt` | 1926 | `356ea0ce45cb250a48fa3ca7fffb3d3447fac61ae2b1b192610e4a3e9a5191fd` |
| `rice_CVFs.csv` | 67957 | `a11b507f294037e34cf91865db6f827fb4003778ab0962f93b949a58d24f4021` |
| `rice_geno.csv` | 50527158 | `04768d911e1cacdd7c4cdbc2160a868a47e1ece2b128bedb0f18993db13205ad` |
| `rice_pheno.csv` | 19797 | `2e34e048122b0ab10c397ec944e34165c2ecbb0aca5830e877f207b98386e325` |
| `sorghum_CVFs.csv` | 94386 | `6577cbb9ff60386f807a0e92ea985b8ae38bd05bb739529bee770065d081135c` |
| `sorghum_geno.csv` | 68305935 | `36c8b89caaa811bbdf1bcffbeec0205b504015a4201690fce70cdb3666588566` |
| `sorghum_pheno.csv` | 28182 | `8f216c625176362b2e740f2f1b7e7d6bb8e6da73b40894653f721eecf00c84a0` |
| `soy_CVFs.csv` | 1058549 | `db9d9ced0d263aa8dad018e7e2de82ebd13f54dc4a18d4185cdf522bedc455cb` |
| `soy_geno.csv` | 50038368 | `635422f193fc96dc48dbe043aac813f388c65daca0638acaeb53f07af98e1a65` |
| `soy_pheno.csv` | 327918 | `33b146712bb3df1e29827d1df34acf9c32b4faa4651008f36a281e68d6d7924a` |
| `spruce_CVFs.csv` | 352604 | `8ee19ea1311fd3e2d21e1b45390a08e73e8422f6c70139d42d4e7bb3fa06acbc` |
| `spruce_geno.csv` | 31867341 | `da1c6cacee24f9f51c35c4d534acf960880baaac463c4af46d7d63265ee1e703` |
| `spruce_pheno.csv` | 101392 | `2a62e5117a598949ca6bfdcf72b43c54b11890b976b688b5e7063a5dedfb44ee` |
| `switchgrass_CVFs.csv` | 109225 | `11980591594b939d70eb6d613c8274263dc2703e2feea275874d588dfb223470` |
| `switchgrass_geno.csv` | 237057149 | `c75dcb4693b4e6efbc888acefd783e5d73ca1bccd0cc48ba8b4530e0d2190197` |
| `switchgrass_pheno.csv` | 33780 | `ab7876444e7805920f8b9b5b2c8d64f1354970f29109f398c43f37bdf05dca56` |

## Files

For each species (`maize`, `rice`, `sorghum`, `soy`, `spruce`, `switchgrass`):

- `<species>_geno.csv`: one row per line (plant variety), one column per SNP marker,
  coded -1/0/1 (homozygous minor / heterozygous / homozygous major allele).
- `<species>_pheno.csv`: three traits per line.
- `<species>_CVFs.csv`: 100 columns (`cv_1`…`cv_100`), each assigning every line to one of
  5 cross-validation folds.
- `README.txt`: the original Dryad readme.

`azodi2019_tableS5.csv` is Table S5 of the paper: the mean Pearson r of 18 genomic
prediction methods on each of the 18 species/trait pairs (324 rows). It is copied from the
paper's supplement with the footnote rows removed.

Rice genotypes include a small fraction (about 3%) of imputed, non-integer dosages; the
other five species are coded strictly -1/0/1.

## Held-out-family folds (added for the manuscript)

`soy_family_folds.csv` has the same layout as `soy_CVFs.csv`, but each of its 10 columns
holds out 8 of the 38 SoyNAM families (fold 1) and trains on the other 30 (fold 2). It is
written by `make_family_folds.py` (fixed random seed) and is passed to the fitting scripts
with `--folds data/soy_family_folds.csv`.
