#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Figure 5 - weight structure & pathway biology:
#           a correlation heatmap | b UMAP / c pathway enrichment.
# Inputs  : results/all_weights.csv, results/disease_summary.csv,
#           results/fgsea_enrichment.csv
# Outputs : output/fig5.{pdf,png} (+ provenance)
# Stage   : L5 manuscript
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(ggrepel)
  library(patchwork); library(scales); library(uwot); library(stringr)
  library(jsonlite)
})

set.seed(42)

BASE        <- normalizePath(file.path(dirname(sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
                 "..", ".."))
RESULTS_DIR <- file.path(BASE, "results")
FIG_DIR     <- file.path(BASE, "output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
cat("BASE:", BASE, "\n")

# --- theme (inlined from house style) ----------------------------------------
theme_n <- theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line  = element_line(linewidth = 0.35),
        axis.ticks = element_line(linewidth = 0.25),
        axis.text  = element_text(size = 6, colour = "black"),
        axis.title = element_text(size = 7),
        plot.title = element_text(face = "bold", size = 7),
        legend.key.size = unit(0.3, "cm"),
        legend.title = element_text(size = 6), legend.text = element_text(size = 6))

# --- constants ---------------------------------------------------------------
GROUP_COLS <- c(
  "Brain / Neurological"="#4E79A7", "Cardiovascular"="#E15759",
  "Metabolic / Organ"="#F28E2B", "Respiratory"="#76B7B2",
  "Cancer"="#59A14F", "Musculoskeletal"="#EDC948",
  "Ophthalmic"="#FF9DA7", "Psychiatric"="#B07AA1")
DL <- c(
  alzheimer_disease="Alzheimer disease", all_cause_dementia="All-cause dementia",
  vascular_dementia="Vascular dementia", frontotemporal_dementia="Frontotemporal dementia",
  parkinson_disease_and_parkinsonism="Parkinson disease",
  amyotrophic_lateral_sclerosis="Amyotrophic lateral sclerosis", depression="Depression",
  ischemic_heart_disease="Ischaemic heart disease", hypertensive_disease="Hypertensive disease",
  heart_failure="Heart failure", atrial_fibrillation_or_flutter="Atrial fibrillation",
  type2_diabetes="Type 2 diabetes", cerebrovascular_disease="Cerebrovascular disease",
  chronic_kidney_disease="Chronic kidney disease", chronic_liver_disease="Chronic liver disease",
  emphysema_copd="COPD / emphysema", colorectal_cancer="Colorectal cancer",
  lung_cancer="Lung cancer", breast_cancer="Breast cancer", prostate_cancer="Prostate cancer",
  leukemia="Leukaemia", non_hodgkin_lymphoma="Non-Hodgkin lymphoma",
  esophageal_cancer="Oesophageal cancer", liver_cancer="Liver cancer",
  pancreatic_cancer="Pancreatic cancer", brain_cancer="Brain cancer",
  ovarian_cancer="Ovarian cancer", rheumatoid_arthritis="Rheumatoid arthritis",
  osteoporosis="Osteoporosis", osteoarthritis="Osteoarthritis",
  macular_degeneration="Macular degeneration")

# --- load --------------------------------------------------------------------
league <- read_csv(file.path(RESULTS_DIR, "disease_summary.csv"),
                   show_col_types = FALSE)
SIG_DISEASES <- league$outcome[league$p_value < 0.05]

W_dis <- read_csv(file.path(RESULTS_DIR, "all_weights.csv"),
                  show_col_types = FALSE)
prot_names <- W_dis[[1]]; W_dis <- W_dis[, -1]
sig_cols <- intersect(SIG_DISEASES, names(W_dis))
W22 <- as.matrix(W_dis[, sig_cols]); rownames(W22) <- prot_names
dis_labels <- DL[sig_cols]
X <- t(W22)
dis_to_cat <- league$group[match(sig_cols, league$outcome)]

# --- Panel a: correlation heatmap --------------------------------------------
cor_orig <- cor(t(X)); rownames(cor_orig) <- dis_labels; colnames(cor_orig) <- dis_labels
hc <- hclust(as.dist(1 - cor_orig), method = "average"); ord <- hc$order
df_heat <- expand.grid(d1 = factor(dis_labels[ord], levels = dis_labels[ord]),
                       d2 = factor(dis_labels[ord], levels = dis_labels[ord]))
df_heat$r <- as.vector(cor_orig[ord, ord])
tick_colors <- unname(GROUP_COLS[dis_to_cat[ord]])
p_a <- ggplot(df_heat, aes(d1, d2, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.25) +
  scale_fill_distiller(palette = "RdYlBu", direction = -1, limits = c(0.15, 1),
                       oob = squish, name = expression(italic(r)), breaks = seq(0.2, 1, .2)) +
  labs(x = NULL, y = NULL, title = "Pairwise weight correlation", tag = "a") + theme_n +
  theme(axis.text.x = element_blank(), axis.text.y = element_text(size = 5, colour = tick_colors),
        axis.ticks = element_blank(), axis.line = element_blank(), aspect.ratio = 1,
        legend.key.height = unit(0.45, "cm"), legend.key.width = unit(0.15, "cm"),
        legend.text = element_text(size = 5), legend.title = element_text(size = 5.5))

# --- Panel b: UMAP -----------------------------------------------------------
set.seed(7)
emb <- umap(X, n_neighbors = 10, min_dist = 0.8, metric = "cosine", n_components = 2)
df_umap <- data.frame(x = emb[, 1], y = emb[, 2], label = dis_labels, cat = dis_to_cat)
pad <- 5.0
hull_u <- do.call(rbind, lapply(split(df_umap, df_umap$cat),
  function(g) if (nrow(g) >= 3) g[chull(g$x, g$y), ] else NULL))
p_b <- ggplot(df_umap, aes(x, y, colour = cat)) +
  geom_polygon(data = hull_u, aes(fill = cat), colour = NA, alpha = 0.10, show.legend = FALSE) +
  geom_point(size = 3.0, colour = "white") +
  geom_point(size = 2.1, shape = 16, alpha = 0.95) +
  scale_colour_manual(values = GROUP_COLS, name = NULL,
                      labels = c("Brain / Neurological" = "Brain", "Cardiovascular" = "Cardiovascular",
                                 "Metabolic / Organ" = "Metabolic", "Respiratory" = "Respiratory",
                                 "Cancer" = "Cancer", "Musculoskeletal" = "Musculoskel.",
                                 "Ophthalmic" = "Ophthalmic", "Psychiatric" = "Psychiatric")) +
  scale_fill_manual(values = GROUP_COLS, guide = "none") +
  geom_text_repel(aes(label = label), size = 1.7, max.overlaps = Inf, segment.size = 0.2,
                  segment.colour = "grey55", box.padding = 0.5, point.padding = 0.3,
                  min.segment.length = 0, seed = 42, force = 20, force_pull = 0.02,
                  max.iter = 10000, bg.color = "white", bg.r = 0.12) +
  coord_cartesian(xlim = c(min(emb[,1])-pad, max(emb[,1])+pad),
                  ylim = c(min(emb[,2])-pad, max(emb[,2])+pad)) +
  labs(x = "UMAP 1", y = "UMAP 2", title = "UMAP of weight vectors", tag = "b") + theme_n +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), legend.position = "bottom",
        aspect.ratio = 1, legend.text = element_text(size = 4.4),
        legend.key.size = unit(0.16, "cm"), legend.margin = margin(t = -5),
        legend.spacing.x = unit(0.04, "cm")) +
  guides(colour = guide_legend(nrow = 2, override.aes = list(size = 1.4)))

# --- Panel c: enrichment dot plot --------------------------------------------
enrich <- read_csv(file.path(RESULTS_DIR, "fgsea_enrichment.csv"),
                   show_col_types = FALSE) %>%
  mutate(component = ifelse(component %in% names(DL), DL[component], component))
dis_enrich <- c("Shared component (PC1)", "Alzheimer disease", "Type 2 diabetes",
                "Heart failure", "Chronic kidney disease", "Rheumatoid arthritis")
top5 <- enrich %>% filter(component %in% dis_enrich) %>% group_by(component) %>%
  arrange(padj, pval) %>% slice_head(n = 5) %>% ungroup()
all_pw <- unique(top5$pathway_clean)
dot_df <- enrich %>% filter(component %in% dis_enrich, pathway_clean %in% all_pw) %>%
  mutate(neg_log_fdr = -log10(padj), sig = padj < 0.05,
         component = factor(component, levels = dis_enrich),
         pathway_short = str_wrap(recode(pathway_clean,
           "Regulation of Insulin Like Growth Factor Igf Transport and Uptake by Insulin Like Growth Factor Binding Proteins Igfbps" =
             "IGF transport and uptake by IGFBPs"), width = 30))
pw_order <- dot_df %>% group_by(pathway_short) %>% slice_min(padj, n = 1) %>% ungroup() %>%
  arrange(component, padj) %>% pull(pathway_short) %>% unique()
dot_df <- dot_df %>% mutate(pathway_short = factor(pathway_short, levels = rev(pw_order)))
p_c <- ggplot(dot_df, aes(y = component, x = pathway_short)) +
  geom_point(aes(size = neg_log_fdr, fill = NES), shape = 21, colour = "grey40", stroke = 0.25) +
  geom_point(data = filter(dot_df, sig), aes(size = neg_log_fdr), shape = 21,
             colour = "black", stroke = 0.5, fill = NA, show.legend = FALSE) +
  scale_size_continuous(range = c(0.8, 4.5), name = "FDR",
                        breaks = c(-log10(0.05), -log10(0.01), -log10(0.001)),
                        labels = c("0.05", "0.01", "0.001")) +
  scale_fill_gradient(low = "#FEE0D2", high = "#CB181D", name = "NES", limits = c(1, NA)) +
  labs(x = NULL, y = NULL, title = "Pathway enrichment", tag = "c") + theme_n +
  theme(axis.text.x = element_text(angle = 55, hjust = 1, size = 5), axis.text.y = element_text(size = 6),
        axis.ticks = element_blank(), axis.line = element_blank(),
        panel.grid.major = element_line(colour = "grey94", linewidth = 0.15),
        legend.key.height = unit(0.35, "cm"), legend.key.width = unit(0.15, "cm"),
        legend.text = element_text(size = 5), legend.title = element_text(size = 5.5),
        legend.position = "right")

# --- assemble & save ---------------------------------------------------------
AB_R_A <- 14.0
AB_L_B <- 5.8
ab_row <- (p_a + theme(plot.margin = margin(2, AB_R_A, 0, 2.5, "mm"))) |
          (p_b + theme(plot.margin = margin(2, 0, 0, AB_L_B, "mm")))

inline_tag <- theme(plot.title.position = "plot",
                    plot.tag.position = c(0, 1),
                    plot.tag = element_text(face = "bold", size = 10, hjust = 0, vjust = 1),
                    plot.title = element_text(face = "bold", size = 7,
                                              margin = margin(b = 3, l = 11)))

fig5 <- (ab_row & inline_tag) / free(p_c & inline_tag, side = "r") +
  plot_layout(heights = c(1, 0.56)) &
  theme(plot.tag = element_text(face = "bold", size = 10))

W_MM <- 183
H_MM <- 130
stem <- file.path(FIG_DIR, "fig5")
ggsave(paste0(stem, ".pdf"), fig5, width = W_MM/25.4, height = H_MM/25.4, device = pdf)
ggsave(paste0(stem, ".png"), fig5, width = W_MM/25.4, height = H_MM/25.4, dpi = 300)

write_json(list(
  figure  = "fig5",
  script  = "figures/fig5/fig5.R",
  date    = format(Sys.Date()),
  panels  = c(a = "pairwise weight correlation heatmap",
              b = "UMAP of weight vectors",
              c = "pathway enrichment"),
  sources = c("results/all_weights.csv", "results/disease_summary.csv",
              "results/fgsea_enrichment.csv"),
  n_diseases   = length(sig_cols),
  n_proteins   = nrow(W22),
  umap_params  = list(seed = 7, n_neighbors = 10, min_dist = 0.8, metric = "cosine"),
  enrich_components = dis_enrich,
  size_mm      = c(W_MM, H_MM),
  R_version    = R.version.string
), paste0(stem, ".provenance.json"), auto_unbox = TRUE, pretty = TRUE)

cat("Saved:", stem, ".pdf/.png\n")
