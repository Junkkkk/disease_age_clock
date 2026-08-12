#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Figure 4 (revised 2026-07-27) - clinical utility of the disease age
#           clock: prevalent OR, incident HR, incidence stratification, and age
#           gap vs time-to-onset.
# Inputs  : results/disease_summary.csv, results/logistic_or.csv,
#           results/cox_hr.csv, results/cumulative_incidence.csv,
#           results/agegap_vs_onset.csv
# Outputs : output/fig4.{pdf,png} (+ provenance)
# Stage   : L5 manuscript
# Depends : fig3_walkthrough.ipynb (panels a-e originally built there and saved as
#           figures/fig3_clinical.*; the figure was later renumbered to Fig 4 and
#           the generator for figures/fig4_revised.* was lost. This script is the
#           reconstruction and is now the authority for this figure.)
#
# Revisions vs figures/fig4_revised.pdf (2026-07-27):
#   panel titles are placed on the same line as the a/b/c/d/e panel tag, matching
#   the convention now used in fig2_revised.R and fig3_revised.R.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(patchwork); library(jsonlite)
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

# Follow-up length is no longer printed in the panel b title (kept for the
# provenance sidecar and the caption). Manuscript body: "median follow-up 10.4
# years; maximum follow-up 17.6 years" - the old figure's unsourced "up to 18-yr"
# is not used.
FOLLOWUP_LAB <- "median follow-up 10.4 yr"

# Panel titles. Wording follows the manuscript body/caption so figure and text
# use the same terms (prevalent / incident / relative vs absolute risk /
# subclinical progression).
TITLES <- c(
  a = "Disease age gap is associated with prevalent disease",
  b = "Disease age gap predicts incident disease",
  c = "Relative risk among same-age peers (disease age gap)",
  d = "Absolute risk (full disease age)",
  # "widens" is avoided: the gap is measured once at baseline and compared
  # between people at different distances from onset, not within a person
  e = "Baseline disease age gap is larger closer to onset"
)

CI_DISEASES <- c("type2_diabetes", "chronic_kidney_disease",
                 "heart_failure", "alzheimer_disease", "rheumatoid_arthritis")

# panel e binning: markers are means of 2.5-yr bins out to 15 yr before onset
BIN_W   <- 2.5
BIN_MIN <- 5
T_MAX   <- 15

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

BIO_LINE_COLS <- c("Old" = "#E15759", "Normal" = "#888888", "Young" = "#4E79A7")

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

DL_SHORT <- c(
  type2_diabetes = "Type 2 diabetes", chronic_kidney_disease = "Chronic kidney disease",
  heart_failure = "Heart failure", alzheimer_disease = "Alzheimer disease",
  rheumatoid_arthritis = "Rheumatoid arthritis"
)

theme_nature <- theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(
    axis.line         = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks        = element_line(linewidth = 0.25, colour = "black"),
    axis.text         = element_text(size = 6, colour = "black"),
    axis.title        = element_text(size = 7, colour = "black"),
    legend.background = element_blank(),
    legend.key        = element_blank(),
    legend.key.size   = unit(0.3, "cm"),
    legend.text       = element_text(size = 5.5),
    legend.title      = element_text(size = 6, face = "bold"),
    # title sits on the same line as the panel tag: anchored to the plot (not the
    # panel) and indented just past the tag glyph
    plot.title.position = "plot",
    plot.title        = element_text(face = "bold", size = 7.5, colour = "black",
                                     hjust = 0, margin = margin(l = 13, b = 3)),
    plot.tag          = element_text(face = "bold", size = 10),
    plot.tag.position = c(0, 1),
    plot.margin       = margin(3, 3, 3, 3),
    strip.text        = element_text(face = "bold", size = 6.5, colour = "black"),
    strip.background  = element_rect(fill = "grey96", colour = NA, linewidth = 0)
  )

# facet panels use open (left + bottom only) axes, no boxed panel border and no
# grey strip background
km_theme <- theme_nature +
  theme(strip.text       = element_text(face = "bold", size = 6),
        strip.background = element_blank(),
        panel.border     = element_blank(),
        axis.line        = element_line(linewidth = 0.35, colour = "black"))

# --- load --------------------------------------------------------------------
league <- read_csv(file.path(RESULTS_DIR, "disease_summary.csv"),
                   show_col_types = FALSE) %>%
  filter(!outcome %in% EXCLUDE) %>%
  mutate(label = DL[outcome], sig = p_value < 0.05)
N_TESTS      <- nrow(league)
SIG_DISEASES <- league %>% filter(sig) %>% pull(outcome)
cat("N_TESTS:", N_TESTS, "  significant:", length(SIG_DISEASES), "\n")

add_meta <- function(path) {
  read_csv(file.path(RESULTS_DIR, path), show_col_types = FALSE) %>%
    filter(outcome %in% SIG_DISEASES) %>%
    mutate(label = DL[outcome],
           group = league$group[match(outcome, league$outcome)])
}
logOR_yr <- add_meta("logistic_or.csv")
cox_yr   <- add_meta("cox_hr.csv")

# --- Panel a: prevalent-case odds ratios -------------------------------------
df4a <- logOR_yr %>%
  filter(!is.na(OR_per_yr)) %>%
  mutate(sig_or = p_M < 0.05 / N_TESTS,
         label  = factor(label, levels = arrange(., OR_per_yr)$label))

p4a <- ggplot(df4a, aes(x = OR_per_yr, y = label)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_errorbar(aes(xmin = OR_lo, xmax = OR_hi, colour = group),
                width = 0, linewidth = 0.5) +
  geom_point(aes(colour = group, shape = sig_or), size = 1.5) +
  scale_colour_manual(values = GROUP_COLS, name = NULL,
                      guide = guide_legend(nrow = 1)) +
  scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 1), guide = "none") +
  labs(x = "OR per year of disease age gap", y = NULL,
       title = TITLES[["a"]]) +
  theme_nature +
  theme(axis.text.y = element_text(size = 5))

# --- Panel b: incident-case hazard ratios ------------------------------------
df4b <- cox_yr %>%
  filter(!is.na(HR_per_yr)) %>%
  mutate(sig_hr = p_M < 0.05 / N_TESTS,
         label  = factor(label, levels = arrange(., HR_per_yr)$label))

p4b <- ggplot(df4b, aes(x = HR_per_yr, y = label)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_errorbar(aes(xmin = HR_lo, xmax = HR_hi, colour = group),
                width = 0, linewidth = 0.5) +
  geom_point(aes(colour = group, shape = sig_hr), size = 1.5) +
  scale_colour_manual(values = GROUP_COLS, guide = "none") +
  scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 1), guide = "none") +
  labs(x = "HR per year of disease age gap", y = NULL,
       title = TITLES[["b"]]) +
  theme_nature +
  theme(axis.text.y = element_text(size = 5))

cat("panel a: n =", nrow(df4a), " Bonferroni sig =", sum(df4a$sig_or),
    " OR range", round(range(df4a$OR_per_yr), 2), "\n")
cat("panel b: n =", nrow(df4b), " Bonferroni sig =", sum(df4b$sig_hr),
    " HR range", round(range(df4b$HR_per_yr), 2), "\n")

# --- Panels c/d: cumulative incidence by stratification ----------------------
ci <- read_csv(file.path(RESULTS_DIR, "cumulative_incidence.csv"),
               show_col_types = FALSE)

ci_subset <- function(strat) {
  ci %>%
    filter(outcome %in% CI_DISEASES, stratification == strat) %>%
    mutate(label     = factor(DL_SHORT[outcome], levels = DL_SHORT[CI_DISEASES]),
           bio_group = factor(bio_group, levels = c("Old", "Normal", "Young")))
}
df_ci_gap <- ci_subset("age_gap")
df_ci_raw <- ci_subset("raw_M")

km_panel <- function(df, title, xlab) {
  ggplot(df, aes(x = time, y = cumulative_incidence,
                 colour = bio_group, fill = bio_group)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.55) +
    facet_wrap(~ label, nrow = 1, scales = "free_y") +
    scale_colour_manual(values = BIO_LINE_COLS, name = NULL,
                        labels = c("Old (top 10%)", "Normal (mid 80%)",
                                   "Young (bottom 10%)")) +
    scale_fill_manual(values = BIO_LINE_COLS, guide = "none") +
    labs(x = xlab, y = "Incidence (%)", title = title) +
    km_theme
}

p4c <- km_panel(df_ci_gap, TITLES[["c"]], NULL) +
  theme(legend.position = "none",
        axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        plot.margin = margin(3, 3, 8, 3))

p4d <- km_panel(df_ci_raw, TITLES[["d"]], "Years from baseline") +
  theme(legend.position = "bottom",
        strip.text = element_blank(),
        legend.margin = margin(t = -3),
        plot.margin = margin(8, 3, 3, 3))

# --- Panel e: age gap vs time to onset ---------------------------------------
df_onset <- read_csv(file.path(RESULTS_DIR, "agegap_vs_onset.csv"),
                     show_col_types = FALSE) %>%
  filter(outcome %in% CI_DISEASES) %>%
  mutate(label = factor(DL_SHORT[outcome], levels = DL_SHORT[CI_DISEASES]),
         # signed time so the axis reads -15 .. 0 with onset at the right,
         # instead of a reversed positive axis
         t_signed = -time_to_onset)

df_onset <- df_onset %>%
  mutate(bin_idx    = floor(time_to_onset / BIN_W),
         bin_centre = -(bin_idx * BIN_W + BIN_W / 2))

slope_pvals <- df_onset %>%
  group_by(label) %>%
  summarise(slope = summary(lm(age_gap ~ time_to_onset))$coefficients[2, 1],
            pval  = summary(lm(age_gap ~ time_to_onset))$coefficients[2, 4],
            .groups = "drop") %>%
  mutate(p_label = ifelse(pval < 1e-10,
                          paste0("p = ", formatC(pval, format = "e", digits = 0)),
                   ifelse(pval < 0.001,
                          paste0("p = ", formatC(pval, format = "e", digits = 1)),
                          paste0("p = ", formatC(pval, format = "f", digits = 3)))))
print(slope_pvals)

# binned summary drawn on top of the cases. Bins stop at 15 yr: the 15-17.5 yr
# bin holds 3-18 cases against 48-340 elsewhere and is the only reason the five
# diseases would carry different numbers of markers. The lm below still uses
# every case, so the slopes and p values are unaffected by the bin definition.
bin_stats <- df_onset %>%
  filter(time_to_onset < T_MAX) %>%
  group_by(label, bin_centre) %>%
  filter(dplyr::n() >= BIN_MIN) %>%
  summarise(n    = dplyr::n(),
            mean = mean(age_gap),
            se   = sd(age_gap) / sqrt(dplyr::n()),
            .groups = "drop") %>%
  mutate(ci_lo = mean - 1.96 * se,
         ci_hi = mean + 1.96 * se)
cat("panel e bins (2.5 yr, n >=", BIN_MIN, "):\n")
print(as.data.frame(bin_stats))

# case counts for the regression, placed opposite the p value
n_lab <- df_onset %>%
  count(label, name = "n") %>%
  mutate(lab = paste0("n = ", format(n, big.mark = ",", trim = TRUE)))

# the window is set from the spread of age gaps, not from the CI bounds: a window
# derived from the CIs opens ~12 yr above the median and only ~4 yr below, which
# clips the cases one-sidedly and makes the visible cloud read high
Y_ZOOM <- c(-6, 15)
off <- df_onset %>%
  filter(time_to_onset < T_MAX) %>%
  group_by(label) %>%
  summarise(n     = dplyr::n(),
            below = round(100 * mean(age_gap < Y_ZOOM[1]), 1),
            above = round(100 * mean(age_gap > Y_ZOOM[2]), 1),
            .groups = "drop")
cat("panel e y window", Y_ZOOM, "- % of cases clipped:\n")
print(as.data.frame(off))

# onset (t = 0) sits at the right. x is fixed across facets: a free x scale would
# make the same drawn angle span a different number of years in each disease, so
# the five slopes could not be compared by eye. Two layers with different sources:
# grey points and the binned means summarise the cases, the red line is an lm fit
# to the individual cases and the p value belongs to that line.
p4e <- ggplot(df_onset, aes(x = t_signed, y = age_gap)) +
  geom_point(colour = "grey78", size = 0.22, stroke = 0, alpha = 0.35) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.25) +
  # fullrange draws the fit across the common -15..0 axis rather than stopping at
  # each disease's own data range, so the slopes are read over one interval. The
  # ribbon widens where a disease has no cases, marking the extrapolated part.
  geom_smooth(method = "lm", se = TRUE, fullrange = TRUE, colour = "#E15759",
              linewidth = 0.6, fill = "#E15759", alpha = 0.15) +
  geom_errorbar(data = bin_stats, inherit.aes = FALSE,
                aes(x = bin_centre, ymin = ci_lo, ymax = ci_hi),
                width = 0, colour = "grey25", linewidth = 0.3) +
  geom_point(data = bin_stats, inherit.aes = FALSE,
             aes(x = bin_centre, y = mean),
             shape = 21, fill = "white", colour = "grey15",
             size = 1.1, stroke = 0.35) +
  geom_text(data = slope_pvals, aes(label = p_label), x = -Inf, y = Inf,
            hjust = -0.05, vjust = 1.4, size = 1.9, fontface = "bold",
            colour = "black", inherit.aes = FALSE) +
  geom_text(data = n_lab, aes(label = lab), x = Inf, y = -Inf,
            hjust = 1.08, vjust = -0.7, size = 1.7, colour = "grey30",
            inherit.aes = FALSE) +
  # ticks every 2.5 yr mark the bin boundaries, so each marker sits midway
  # between two ticks; only every second tick is labelled to avoid crowding
  scale_x_continuous(breaks = seq(-T_MAX, 0, BIN_W),
                     labels = ifelse(seq(-T_MAX, 0, BIN_W) %% 5 == 0,
                                     as.character(seq(-T_MAX, 0, BIN_W)), "")) +
  # explicit 5-yr breaks, matching ED Fig 2 so both panels read on the same scale
  scale_y_continuous(breaks = seq(-5, 15, 5)) +
  facet_wrap(~ label, nrow = 1) +
  # display window only: coord_cartesian clips the display, it does not drop rows,
  # so the regression and its CI are still fit on the full data. The clipped share
  # is printed above and belongs in the caption.
  coord_cartesian(xlim = c(-T_MAX - 0.3, 0.3), ylim = Y_ZOOM) +
  labs(x = "Time to disease onset (years)", y = "Disease age gap (years)",
       title = TITLES[["e"]]) +
  km_theme

# --- assemble & save ---------------------------------------------------------
# the group colour key is collected once for the a|b row; c/d keep their own
# stratification key and e needs none
top_row <- (p4a | p4b) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.2, "cm"),
        legend.margin = margin(t = -2))

# free(side = "l"): release the c/d and e rows from patchwork's column alignment
# with the long disease labels of panels a/b, so they reach the figure's left edge
fig4 <- (top_row / free(p4c / p4d, side = "l") / free(p4e, side = "l")) +
  plot_layout(heights = c(1, 0.85, 0.42)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 10, hjust = 0, vjust = 1),
        plot.tag.position = c(0, 1))

W_MM <- 183
H_MM <- 230
stem <- file.path(FIG_DIR, "fig4")
ggsave(paste0(stem, ".pdf"), fig4, width = W_MM/25.4, height = H_MM/25.4, device = "pdf")
ggsave(paste0(stem, ".png"), fig4, width = W_MM/25.4, height = H_MM/25.4, dpi = 300)

write_json(list(
  figure  = "fig4",
  script  = "figures/fig4/fig4.R",
  date    = format(Sys.Date()),
  sources = c("results/disease_summary.csv", "results/logistic_or.csv",
              "results/cox_hr.csv", "results/cumulative_incidence.csv",
              "results/agegap_vs_onset.csv"),
  n_tests            = N_TESTS,
  n_diseases_panel_a = nrow(df4a),
  n_diseases_panel_b = nrow(df4b),
  n_bonferroni_a     = sum(df4a$sig_or),
  n_bonferroni_b     = sum(df4b$sig_hr),
  ci_diseases        = unname(DL_SHORT[CI_DISEASES]),
  followup_label     = FOLLOWUP_LAB,
  onset_slopes       = slope_pvals %>% mutate(label = as.character(label)),
  panel_e_bin_width  = BIN_W,
  panel_e_bin_min_n  = BIN_MIN,
  panel_e_bin_max_yr = T_MAX,
  panel_e_y_window   = Y_ZOOM,
  panel_e_bin_stats  = bin_stats %>% mutate(label = as.character(label)),
  panel_e_clipping   = off %>% mutate(label = as.character(label)),
  panel_e_note       = paste("grey points = individual incident cases;",
                             "open circles = mean age gap within 2.5-yr bins with",
                             "95% CI (mean +/- 1.96 SE), bins out to 15 yr before",
                             "onset and >= 5 cases; red line = lm of age gap on",
                             "time to onset fit to all cases, p value refers to",
                             "the line. Bin means and the fit use every case;",
                             "the y window clips the display only."),
  R_version          = R.version.string
), paste0(stem, ".provenance.json"), auto_unbox = TRUE, pretty = TRUE)

cat("Saved:", stem, ".pdf/.png\n")
