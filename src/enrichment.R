#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Pathway enrichment analysis (fgsea) on disease clock weights
# Inputs  : results/all_weights.csv, results/disease_summary.csv
# Outputs : results/fgsea_enrichment.csv
# Stage   : L4 analysis (between clock construction and Figure 5)
# Depends : src/ukb_analysis.py (produces all_weights.csv, disease_summary.csv)
#
# For each disease clock (+ the shared PC1 component), ranks proteins by
# absolute weight and runs fgseaMultilevel against Hallmark, KEGG, Reactome,
# and GO:BP pathway databases from msigdbr.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(fgsea); library(msigdbr)
})

set.seed(42)

BASE        <- normalizePath(file.path(dirname(sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
                 ".."))
RESULTS_DIR <- file.path(BASE, "results")
cat("BASE:", BASE, "\n")

# --- load weights & significance ---------------------------------------------
W <- read_csv(file.path(RESULTS_DIR, "all_weights.csv"), show_col_types = FALSE)
proteins <- toupper(W$protein)
W <- W[, -1]
cat("Proteins:", length(proteins), " Diseases:", ncol(W), "\n")

league <- read_csv(file.path(RESULTS_DIR, "disease_summary.csv"),
                   show_col_types = FALSE)

# --- pathway databases -------------------------------------------------------
cat("Loading pathway databases...\n")
hallmark_pw <- split(toupper(msigdbr(species="Homo sapiens", collection="H")$gene_symbol),
                     msigdbr(species="Homo sapiens", collection="H")$gs_name)
kegg_pw     <- split(toupper(msigdbr(species="Homo sapiens", collection="C2", subcollection="CP:KEGG_LEGACY")$gene_symbol),
                     msigdbr(species="Homo sapiens", collection="C2", subcollection="CP:KEGG_LEGACY")$gs_name)
reactome_pw <- split(toupper(msigdbr(species="Homo sapiens", collection="C2", subcollection="CP:REACTOME")$gene_symbol),
                     msigdbr(species="Homo sapiens", collection="C2", subcollection="CP:REACTOME")$gs_name)
gobp_pw     <- split(toupper(msigdbr(species="Homo sapiens", collection="C5", subcollection="GO:BP")$gene_symbol),
                     msigdbr(species="Homo sapiens", collection="C5", subcollection="GO:BP")$gs_name)

# --- fgsea per component -----------------------------------------------------
run_fgsea_enrich <- function(w, prots, comp_name) {
  set.seed(42)
  stats <- setNames(abs(w), prots)
  stats <- sort(stats, decreasing = TRUE)
  res_list <- list()
  for (db in c("Hallmark", "KEGG", "Reactome", "GO:BP")) {
    pw <- switch(db, Hallmark=hallmark_pw, KEGG=kegg_pw,
                 Reactome=reactome_pw, `GO:BP`=gobp_pw)
    res <- fgseaMultilevel(pathways=pw, stats=stats,
                           minSize=10, maxSize=500, eps=0, scoreType="pos")
    res$database  <- db
    res$component <- comp_name
    res_list[[db]] <- res
  }
  bind_rows(res_list)
}

# shared component: PC1 of significant disease weight matrix
sig_cols <- intersect(league$outcome[league$p_value < 0.05], names(W))
W_sig <- as.matrix(W[, sig_cols])
sv <- svd(t(W_sig))
pc1 <- sv$v[, 1]

cat("Running PC1 (shared component)...")
all_res <- run_fgsea_enrich(pc1, proteins, "Shared component (PC1)")
cat(" done\n")

# per-disease weights
for (cc in names(W)) {
  cat(cc, "...")
  res_cc <- run_fgsea_enrich(W[[cc]], proteins, cc)
  all_res <- bind_rows(all_res, res_cc)
  cat(" done\n")
}

# --- clean & save -------------------------------------------------------------
all_res <- all_res %>%
  mutate(pathway_clean = gsub("^HALLMARK_|^KEGG_|^REACTOME_|^GOBP_", "", pathway),
         pathway_clean = gsub("_", " ", pathway_clean),
         pathway_clean = tools::toTitleCase(tolower(pathway_clean)))

write_csv(
  all_res %>% select(database, component, pathway, pathway_clean, pval, padj, NES, size),
  file.path(RESULTS_DIR, "fgsea_enrichment.csv")
)
cat("\nSaved fgsea_enrichment.csv:", nrow(all_res), "rows\n")
