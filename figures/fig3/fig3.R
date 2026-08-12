#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Figure 3 (revised 2026-07-27) - disease aging clock vs existing clocks
# Inputs  : results/disease_summary.csv, results/fair_comparison.csv
# Outputs : output/fig3.{pdf,png} (+ provenance)
# Stage   : L5 manuscript
# Depends : fig2_walkthrough_R.ipynb (panels a-d originally built there and saved
#           as figures/fig2_atlas.*; the figure was later renumbered to Fig 3 and
#           the generator for figures/fig3_revised.* was lost. This script is the
#           reconstruction and is now the authority for this figure.)
#
# Revisions vs fig3_revised.pdf (user, 2026-07-27):
#   the method name "MaxIE" is not used in any figure label; it is replaced by
#   "Disease aging clock" throughout. Canonical column names in
#   results/fair_comparison.csv (MaxIE_f_star, r_age_maxie, auc_full_maxie) are
#   left untouched - only display text changes.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(ggrepel); library(patchwork); library(forcats)
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

# --- constants ---------------------------------------------------------------
EXCLUDE <- c("fluid_intelligence", "reaction_time")

# display name for our method; "MaxIE" is deliberately not used in labels
CLOCK_LAB <- "Disease aging clock"
ORGAN_LAB <- "Best organ clock"
METHOD_COLS <- setNames(c("#E15759", "#4E79A7"), c(CLOCK_LAB, ORGAN_LAB))

GROUP_COLS <- c(
  "Brain / Neurological"  = "#4E79A7",
  "Cardiovascular"        = "#E15759",
  "Metabolic / Organ"     = "#F28E2B",
  "Respiratory"           = "#76B7B2",
  "Cancer"                = "#59A14F",
  "Musculoskeletal"       = "#EDC948",
  "Ophthalmic"            = "#FF9DA7",
  "Psychiatric"           = "#B07AA1"
)

DL <- c(
  alzheimer_disease="Alzheimer disease", all_cause_dementia="All-cause dementia",
  vascular_dementia="Vascular dementia", frontotemporal_dementia="Frontotemporal dementia",
  parkinson_disease_and_parkinsonism="Parkinson disease",
  amyotrophic_lateral_sclerosis="Amyotrophic lateral sclerosis",
  depression="Depression", ischemic_heart_disease="Ischaemic heart disease",
  hypertensive_disease="Hypertensive disease", heart_failure="Heart failure",
  atrial_fibrillation_or_flutter="Atrial fibrillation", type2_diabetes="Type 2 diabetes",
  cerebrovascular_disease="Cerebrovascular disease",
  chronic_kidney_disease="Chronic kidney disease",
  chronic_liver_disease="Chronic liver disease", emphysema_copd="COPD / emphysema",
  colorectal_cancer="Colorectal cancer", lung_cancer="Lung cancer",
  breast_cancer="Breast cancer", prostate_cancer="Prostate cancer",
  leukemia="Leukaemia", non_hodgkin_lymphoma="Non-Hodgkin lymphoma",
  esophageal_cancer="Oesophageal cancer", liver_cancer="Liver cancer",
  pancreatic_cancer="Pancreatic cancer", brain_cancer="Brain cancer",
  ovarian_cancer="Ovarian cancer", rheumatoid_arthritis="Rheumatoid arthritis",
  osteoporosis="Osteoporosis", osteoarthritis="Osteoarthritis",
  macular_degeneration="Macular degeneration"
)

theme_nature <- theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(
    axis.line         = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks        = element_line(linewidth = 0.25, colour = "black"),
    axis.text         = element_text(size = 6, colour = "black"),
    axis.title        = element_text(size = 7, colour = "black"),
    legend.key        = element_blank(),
    legend.key.size   = unit(0.3, "cm"),
    legend.text       = element_text(size = 5.5),
    legend.title      = element_text(size = 6, face = "bold"),
    # title sits on the same line as the panel tag: anchored to the plot (not the
    # panel) and indented just past the tag glyph
    plot.title.position = "plot",
    plot.title        = element_text(face = "bold", size = 7.5, hjust = 0,
                                     margin = margin(l = 13, b = 4)),
    plot.tag          = element_text(face = "bold", size = 10),
    plot.tag.position = c(0, 1),
    strip.text        = element_text(face = "bold", size = 6.5),
    strip.background  = element_rect(fill = "grey96", colour = NA)
  )

# --- load --------------------------------------------------------------------
league <- read_csv(file.path(RESULTS_DIR, "disease_summary.csv"),
                   show_col_types = FALSE) %>%
  filter(!outcome %in% EXCLUDE) %>%
  mutate(label = DL[outcome], sig = p_value < 0.05)
SIG_DISEASES <- league %>% filter(sig) %>% pull(outcome)

comp <- read_csv(file.path(RESULTS_DIR, "fair_comparison.csv"),
                 show_col_types = FALSE) %>%
  filter(!outcome %in% EXCLUDE) %>%
  mutate(label = DL[outcome])

league_sig  <- league %>% filter(outcome %in% SIG_DISEASES)
comp_sig    <- comp   %>% filter(outcome %in% SIG_DISEASES)
fstar_order <- league_sig %>% arrange(MaxIE_f_star) %>% pull(label)
cat("league_sig rows:", nrow(league_sig), "  comp_sig rows:", nrow(comp_sig), "\n")

# --- Panel a: mediation index, paired dots -----------------------------------
df3a <- league_sig %>%
  select(outcome, label, group, MaxIE_f_star, Organ_f_star) %>%
  pivot_longer(c(MaxIE_f_star, Organ_f_star),
               names_to = "method", values_to = "fstar") %>%
  mutate(method = ifelse(grepl("MaxIE", method), CLOCK_LAB, ORGAN_LAB),
         method = factor(method, levels = c(CLOCK_LAB, ORGAN_LAB)),
         label  = factor(label, levels = fstar_order))

wins_fstar <- league_sig %>%
  summarise(w = sum(MaxIE_f_star > Organ_f_star)) %>% pull(w)
cat("disease clock wins f*:", wins_fstar, "/", nrow(league_sig), "\n")

p3a <- ggplot(df3a, aes(x = fstar, y = label, colour = method)) +
  geom_line(aes(group = label), colour = "grey82", linewidth = 0.3) +
  geom_point(size = 2.2, shape = 16) +
  scale_colour_manual(values = METHOD_COLS, name = NULL) +
  labs(x = expression(Mediation~index~(italic(f) * "*")), y = NULL,
       title = paste0(CLOCK_LAB, " vs best organ clock")) +
  theme_nature +
  theme(axis.text.y = element_text(size = 7),
        plot.margin = margin(t = 3, r = 6, b = 3, l = 3),
        legend.position = c(0.72, 0.06),
        legend.direction = "horizontal",
        legend.background = element_rect(fill = "white", colour = NA))

# --- Panel b: correlation with chronological age -----------------------------
df3b <- comp_sig %>%
  select(outcome, label, group, r_age_maxie, r_age_organ) %>%
  pivot_longer(c(r_age_maxie, r_age_organ),
               names_to = "method", values_to = "r_age") %>%
  mutate(method = ifelse(grepl("maxie", method), CLOCK_LAB, ORGAN_LAB),
         method = factor(method, levels = c(CLOCK_LAB, ORGAN_LAB)),
         label  = factor(label,
                         levels = comp_sig %>% arrange(r_age_organ) %>% pull(label)))

p3b <- ggplot(df3b, aes(x = r_age, y = label, colour = method)) +
  geom_line(aes(group = label), colour = "grey82", linewidth = 0.3) +
  geom_point(size = 2.2, shape = 16) +
  scale_colour_manual(values = METHOD_COLS, guide = "none") +
  labs(x = expression(italic(r) ~ "(score, chronological age)"), y = NULL,
       title = "Correlation with chronological age") +
  theme_nature +
  theme(axis.text.y = element_text(size = 7),
        plot.margin = margin(t = 3, r = 6, b = 3, l = 3))

# --- Panels c/d: AUC scatters ------------------------------------------------
wins_age   <- sum(comp_sig$auc_full_maxie > comp_sig$auc_age, na.rm = TRUE)
wins_organ <- sum(comp_sig$auc_full_maxie > comp_sig$auc_full_organ, na.rm = TRUE)
cat("disease clock wins AUC vs age:", wins_age, "/", nrow(comp_sig),
    " vs organ:", wins_organ, "/", nrow(comp_sig), "\n")

auc_scatter <- function(xvar, xlab, wins, lims) {
  ggplot(comp_sig, aes(x = .data[[xvar]], y = auc_full_maxie, colour = group)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey60", linewidth = 0.4) +
    geom_point(size = 2.2, shape = 16, alpha = 0.85) +
    # repel is given room to work: labels are pushed outward from the crowded
    # diagonal, so no fixed aspect ratio here (coord_fixed squeezes the panel and
    # forces overlaps). ratio is kept 1:1 via the saved panel geometry instead.
    geom_text_repel(aes(label = label), size = 2.0, max.overlaps = Inf,
                    segment.size = 0.15, segment.colour = "grey65",
                    box.padding = 0.45, point.padding = 0.25,
                    force = 6, force_pull = 0.4, max.iter = 20000,
                    min.segment.length = 0, seed = 42) +
    scale_colour_manual(values = GROUP_COLS, name = NULL) +
    coord_cartesian(xlim = lims, ylim = lims, clip = "off") +
    labs(x = xlab, y = paste0(CLOCK_LAB, " AUC"),
         title = paste0(CLOCK_LAB, " wins ", wins, "/", nrow(comp_sig))) +
    theme_nature +
    theme(plot.margin = margin(t = 3, r = 8, b = 3, l = 3)) +
    guides(colour = guide_legend(nrow = 1, byrow = TRUE))
}

p3c <- auc_scatter("auc_age",        "Chronological age AUC", wins_age,   c(0.50, 0.88))
p3d <- auc_scatter("auc_full_organ", "Best organ clock AUC",  wins_organ, c(0.54, 0.88))

# --- assemble & save ---------------------------------------------------------
# guides are collected on the bottom row only, so panel a keeps its own
# two-colour method key inside the panel
bottom_row <- (p3c | p3d) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# free(side = "l") releases the bottom row from column alignment with the long
# disease labels of panels a/b, so c/d start at the figure's left edge
top_row <- (p3a | p3b) + plot_layout(widths = c(1, 1))

fig3 <- (top_row / free(bottom_row, side = "l")) +
  plot_layout(heights = c(1.05, 0.95)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 10, hjust = 0, vjust = 1),
        plot.tag.position = c(0, 1))

W_MM <- 183
H_MM <- 215
stem <- file.path(FIG_DIR, "fig3")
ggsave(paste0(stem, ".pdf"), fig3, width = W_MM/25.4, height = H_MM/25.4, device = "pdf")
ggsave(paste0(stem, ".png"), fig3, width = W_MM/25.4, height = H_MM/25.4, dpi = 300)

write_json(list(
  figure  = "fig3",
  script  = "figures/fig3/fig3.R",
  date    = format(Sys.Date()),
  sources = c("results/disease_summary.csv", "results/fair_comparison.csv"),
  n_diseases          = nrow(comp_sig),
  wins_mediation_index = wins_fstar,
  wins_auc_vs_age      = wins_age,
  wins_auc_vs_organ    = wins_organ,
  method_label         = CLOCK_LAB,
  R_version            = R.version.string
), paste0(stem, ".provenance.json"), auto_unbox = TRUE, pretty = TRUE)

cat("Saved:", stem, ".pdf/.png\n")
