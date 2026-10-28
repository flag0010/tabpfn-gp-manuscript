# rrBLUP (GBLUP via kin.blup) on 10 replicate 80/20 holdouts, mirroring
# Azodi et al. 2019 predict_rrBLUP.R but scoring only the held-out 20%.
# Replicate i uses column cv_i of the CVFs file; lines in fold TEST_FOLD are the test set.
#
# Usage: Rscript rrblup_holdout.R <species> [n_reps] [test_fold]

library(rrBLUP)

args <- commandArgs(trailingOnly = TRUE)
species <- args[1]
n_reps <- if (length(args) >= 2) as.integer(args[2]) else 10
test_fold <- if (length(args) >= 3) as.integer(args[3]) else 1

data_dir <- "data"
X <- as.matrix(read.csv(file.path(data_dir, paste0(species, "_geno.csv")), row.names = 1))
Y <- read.csv(file.path(data_dir, paste0(species, "_pheno.csv")), row.names = 1)
cvs <- read.csv(file.path(data_dir, paste0(species, "_CVFs.csv")), row.names = 1)
stopifnot(identical(rownames(X), rownames(Y)), identical(rownames(X), rownames(cvs)))

# Relationship matrix built exactly as in the authors' script
M <- tcrossprod(scale(X))
M <- M / mean(diag(M))
rownames(M) <- colnames(M) <- 1:nrow(X)

out_path <- file.path("results", paste0(species, "_rrBLUP.csv"))
# Resume: keep finished fits and skip them
out <- if (file.exists(out_path)) read.csv(out_path) else data.frame()
for (trait in names(Y)) {
  y <- Y[[trait]]
  for (rep in 1:n_reps) {
    if (nrow(out) > 0 && any(out$trait == trait & out$rep == rep)) next
    t0 <- Sys.time()
    test <- which(cvs[[paste0("cv_", rep)]] == test_fold)
    yNA <- y
    yNA[test] <- NA
    fit <- kin.blup(data.frame(y = yNA, gid = 1:nrow(X)), K = M, geno = "gid", pheno = "y")
    r <- cor(y[test], fit$g[test])
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    out <- rbind(out, data.frame(species, trait, rep, model = "rrBLUP",
                                 n_train = nrow(X) - length(test), n_test = length(test),
                                 r = r, seconds = secs))
    cat(sprintf("%s %s rep%d r=%.4f (%.1fs)\n", species, trait, rep, r, secs))
    # Write after every fit so an interrupted run keeps finished results
    write.csv(out, out_path, row.names = FALSE)
  }
}
