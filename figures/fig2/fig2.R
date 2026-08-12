#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Figure 2 (revised 2026-07-27) - disease-specific aging models
# Inputs  : results/disease_summary.csv, results/bioage_scores.csv,
#           results/sex_adjusted_gaps.csv, data/case_control_31diseases.csv
# Outputs : output/fig2*.{pdf,png} (+ provenance)
# Stage   : L5 manuscript
# Depends : fig2_revised_walkthrough_R.ipynb (original, 2026-05-12)
#
# Revisions vs original (advisor/user, 2026-07-27):
#   a  title kept as-is ("Cosine test for disease-specific aging")
#   b  title -> "Year-calibrated disease age (Type 2 diabetes)"
#   b  y-axis "MaxIE biological age" -> "Disease age" (MaxIE dropped)
#   c  "personalised" -> "personalized"
#   d  disease order matched to panel a; three layout variants produced:
#        d1 enlarged boxplot | d2 dumbbell (median case vs control)
#        d4 violin + box (raincloud-style, no ggdist dependency)
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(ggrepel); library(patchwork); library(forcats)
  library(hexbin); library(jsonlite)
})

set.seed(1)

BASE        <- normalizePath(file.path(dirname(sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
                 "..", ".."))
DATA_DIR    <- file.path(BASE, "data")
RESULTS_DIR <- file.path(BASE, "results")
FIG_DIR     <- file.path(BASE, "output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
cat("BASE:", BASE, "\n")

# --- constants ---------------------------------------------------------------
EXCLUDE <- c("fluid_intelligence", "reaction_time")

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
SEX_COLS    <- c("Female" = "#E15759", "Male" = "#4E79A7")
STATUS_COLS <- c(Control = "grey88", Case = "#E15759")

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

# --- config: raw data path (not distributed; see README) --------------------
# case_control_31diseases.csv is a per-subject binary case/control matrix
# generated from health outcome data. It is NOT included in this repo.
# Users with data access must supply their own file and set the path below.
CASE_CONTROL_FILE <- file.path(DATA_DIR, "case_control_31diseases.csv")

# --- load --------------------------------------------------------------------
league <- read_csv(file.path(RESULTS_DIR, "disease_summary.csv"),
                   show_col_types = FALSE) %>%
  filter(!outcome %in% EXCLUDE) %>%
  mutate(label = DL[outcome], sig = p_value < 0.05)

N_TESTS      <- nrow(league)
SIG_DISEASES <- league %>% filter(sig) %>% pull(outcome)
cat("league rows:", N_TESTS, "  sig:", length(SIG_DISEASES), "\n")

bioage <- read_csv(file.path(RESULTS_DIR, "bioage_scores.csv"), show_col_types = FALSE)
sex_gaps_all <- read_csv(file.path(RESULTS_DIR, "sex_adjusted_gaps.csv"),
                         show_col_types = FALSE) %>%
  filter(!outcome %in% EXCLUDE) %>%
  mutate(sex_label = ifelse(sex == 0, "Female", "Male"))

# --- Panel a -----------------------------------------------------------------
bonf_thresh <- 0.05 / N_TESTS

df2a <- league %>%
  mutate(neg_log10_p = -log10(p_value)) %>%
  arrange(neg_log10_p) %>%
  mutate(label = factor(label, levels = label))

# panel-a display order: bottom -> top on the y axis (ascending significance)
A_ORDER_ASC <- levels(df2a$label)

p2a <- ggplot(df2a, aes(x = neg_log10_p, y = label, fill = group)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_vline(xintercept = -log10(0.05), linetype = "dashed",
             colour = "grey65", linewidth = 0.3) +
  geom_vline(xintercept = -log10(bonf_thresh), linetype = "dotted",
             colour = "black", linewidth = 0.45) +
  annotate("text", x = -log10(0.05) + 0.12, y = 1.5, label = "p = 0.05",
           size = 2.2, hjust = 0, colour = "grey45", fontface = "italic") +
  annotate("text", x = -log10(bonf_thresh) + 0.12, y = 3.8,
           label = "Bonferroni (p = 0.0016)",
           size = 2.2, hjust = 0, colour = "black", fontface = "italic") +
  scale_fill_manual(values = GROUP_COLS, name = NULL) +
  labs(x = expression(-log[10](italic(p))), y = NULL,
       title = "Cosine test for disease-specific aging") +
  theme_nature +
  theme(legend.position = c(0.98, 0.02),
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.18, "cm"),
        legend.spacing.y = unit(0.05, "cm")) +
  guides(fill = guide_legend(ncol = 2, byrow = TRUE))

# --- Panel b -----------------------------------------------------------------
sex_gaps <- sex_gaps_all %>%
  filter(outcome == "type2_diabetes") %>%
  mutate(predicted_age_sex = chrono_age + age_gap_sex_adj)

bioage_t2d <- bioage %>% filter(outcome == "type2_diabetes") %>%
  left_join(sex_gaps %>% select(IID, sex_label, age_gap_sex_adj, predicted_age_sex),
            by = "IID")

lm_fit  <- lm(predicted_age_sex ~ chrono_age, data = bioage_t2d)
slope_v <- round(coef(lm_fit)[2], 2)
r_val   <- round(cor(bioage_t2d$chrono_age, bioage_t2d$predicted_age_sex,
                     use = "complete.obs"), 2)
cat("r =", r_val, "  calibration slope =", slope_v, "\n")

ax_lim <- c(37, 82)

# dense jittered scatter, mirroring Fig 1b: points coloured by z-scored age gap.
# chronological age is recorded in whole years, so a +-0.5 yr horizontal jitter
# is applied to break the integer column artefact (y values are continuous).
# Points shaded by local 2D point density: dark where dense, fading to light
# grey in the sparse tails. The ramp stops at grey20 rather than pure black so
# the black regression line stays readable through the high-density core.
# Chronological age is whole-year, so x is jittered by +-0.5 yr before the
# density is evaluated (both use the same coordinates).
b_df <- bioage_t2d %>%
  filter(!is.na(chrono_age), !is.na(predicted_age_sex)) %>%
  mutate(x_jit = chrono_age + runif(dplyr::n(), -0.5, 0.5))

b_df$dens_col <- densCols(
  b_df$x_jit, b_df$predicted_age_sex, nbin = 256,
  colramp = colorRampPalette(c("grey93", "grey75", "grey45", "grey20"))
)

p2b <- ggplot(b_df, aes(x = x_jit, y = predicted_age_sex)) +
  geom_point(colour = b_df$dens_col, size = 0.45, stroke = 0) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey45", linewidth = 0.4) +
  geom_smooth(method = "lm", se = FALSE, colour = "black", linewidth = 0.7) +
  scale_x_continuous(limits = ax_lim, expand = c(0, 0)) +
  scale_y_continuous(limits = ax_lim, expand = c(0, 0)) +
  annotate("text", x = 39, y = 79,
           label = paste0("r = ", r_val, ",  slope = ", slope_v),
           size = 2.5, hjust = 0) +
  labs(x = "Chronological age (years)",
       y = "Disease age (years)",
       title = "Year-calibrated disease age (Type 2 diabetes)") +
  theme_nature + coord_fixed() +
  theme(plot.margin = margin(t = 3, r = 10, b = 3, l = 3))

# NOTE (2026-07-27): a sex-split version of panel b (Pooled/Female/Male slopes,
# mirroring the organ-clock calibration panel) was tried and dropped. panel b is
# built on age_gap_sex_adj, so the sex effect is removed by construction; on the
# raw predicted_age the slopes are 0.978 (F) vs 0.980 (M) and mean gaps 0.031 vs
# 0.004 yr, so all three lines coincide. Sex is already covered by panel c.

p2b_hex <- ggplot(bioage_t2d, aes(x = chrono_age, y = predicted_age_sex)) +
  geom_hex(bins = 60) +
  scale_fill_gradient(low = "grey96", high = "grey65", trans = "log10", guide = "none") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey55", linewidth = 0.5) +
  geom_smooth(method = "lm", se = FALSE, colour = "black", linewidth = 0.8) +
  scale_x_continuous(limits = ax_lim, expand = c(0, 0)) +
  scale_y_continuous(limits = ax_lim, expand = c(0, 0)) +
  annotate("text", x = 39, y = 79,
           label = paste0("r = ", r_val, ",  slope = ", slope_v),
           size = 2.5, hjust = 0) +
  labs(x = "Chronological age (years)",
       y = "Disease age (years)",
       title = "Year-calibrated disease age (Type 2 diabetes)") +
  theme_nature + coord_fixed()

# --- Panel c -----------------------------------------------------------------
p2c <- ggplot(bioage_t2d %>% filter(!is.na(sex_label)),
              aes(x = age_gap_sex_adj, fill = sex_label, colour = sex_label)) +
  geom_density(alpha = 0.35, linewidth = 0.5, key_glyph = "path") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.35) +
  scale_fill_manual(values = SEX_COLS, name = NULL) +
  scale_colour_manual(values = SEX_COLS, name = NULL) +
  coord_cartesian(xlim = c(-12, 18)) +
  labs(x = "Disease age gap (years)", y = "Density",
       title = "Sex-personalized calibration (Type 2 diabetes)") +
  theme_nature +
  theme(legend.position = c(0.88, 0.88),
        plot.margin = margin(t = 3, r = 10, b = 3, l = 3))

# --- Panel d data ------------------------------------------------------------
cc <- read_csv(CASE_CONTROL_FILE,
               show_col_types = FALSE) %>% rename(IID = eid)

box_dfs <- list()
for (dis in SIG_DISEASES) {
  ba <- sex_gaps_all %>% filter(outcome == dis) %>% select(IID, age_gap_sex_adj)
  if (dis %in% names(cc)) {
    y_df <- cc[, c("IID", dis)]; names(y_df)[2] <- "Y"
    merged <- ba %>% inner_join(y_df, by = "IID") %>% filter(!is.na(Y))
    merged$disease <- DL[dis]
    merged$status  <- ifelse(merged$Y == 1, "Case", "Control")
    box_dfs[[dis]] <- merged[, c("IID", "disease", "status", "age_gap_sex_adj")]
  }
}
box_all <- bind_rows(box_dfs)

# disease order inherited from panel a (most significant first, left to right)
D_ORDER_DESC <- rev(A_ORDER_ASC)[rev(A_ORDER_ASC) %in% unique(box_all$disease)]
D_ORDER_ASC  <- A_ORDER_ASC[A_ORDER_ASC %in% unique(box_all$disease)]

box_all <- box_all %>%
  mutate(disease_x = factor(disease, levels = D_ORDER_DESC),
         disease_y = factor(disease, levels = D_ORDER_ASC),
         status    = factor(status, levels = c("Control", "Case")))
cat("box_all rows:", nrow(box_all), "  diseases:", nlevels(box_all$disease_x), "\n")

# Shared theme for the full-width bottom panel. Combined with free(side = "l")
# at assembly time this lets the plotting region reach the left edge of the
# figure instead of aligning to panel a's plotting region.
theme_d_wide <- theme(
  axis.text.x       = element_text(angle = 50, hjust = 1, size = 7),
  axis.text.y       = element_text(size = 7),
  axis.title.y      = element_text(margin = margin(r = 1)),
  # left margin only wide enough for the first rotated x label to not clip
  plot.margin       = margin(t = 5, r = 2, b = 5, l = 14),
  plot.title        = element_text(face = "bold", size = 7.5, hjust = 0,
                                   margin = margin(l = 13, b = 6)),
  legend.position   = c(0.005, 0.99),
  legend.justification = c(0, 1),
  legend.direction  = "horizontal",
  legend.key.size   = unit(0.3, "cm"),
  legend.text       = element_text(size = 6.5)
)

# --- Panel d1: enlarged boxplot ----------------------------------------------
p2d1 <- ggplot(box_all, aes(x = disease_x, y = age_gap_sex_adj, fill = status)) +
  geom_boxplot(width = 0.75, outlier.size = 0.08, outlier.alpha = 0.08,
               linewidth = 0.22, outlier.shape = 16,
               position = position_dodge(width = 0.82)) +
  scale_fill_manual(values = STATUS_COLS, name = NULL,
                    breaks = c("Case", "Control")) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  coord_cartesian(ylim = c(-15, 20)) +
  scale_x_discrete(expand = expansion(add = 0.55)) +
  labs(x = NULL, y = "Disease age gap (years)",
       title = "Cases have a larger disease age gap than controls") +
  theme_nature + theme_d_wide +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))

# --- Panel d2: dumbbell (median case vs control) -----------------------------
dumb <- box_all %>%
  group_by(disease_y, status) %>%
  summarise(med = median(age_gap_sex_adj), .groups = "drop") %>%
  pivot_wider(names_from = status, values_from = med) %>%
  mutate(delta = Case - Control)

p2d2 <- ggplot(dumb, aes(y = disease_y)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  geom_segment(aes(x = Control, xend = Case, yend = disease_y),
               colour = "grey60", linewidth = 0.45) +
  geom_point(aes(x = Control, fill = "Control"), shape = 21, size = 1.8,
             colour = "grey35", stroke = 0.25) +
  geom_point(aes(x = Case, fill = "Case"), shape = 21, size = 1.8,
             colour = "grey20", stroke = 0.25) +
  geom_text(aes(x = pmax(Case, Control) + 0.25,
                label = sprintf("+%.1f", delta)),
            size = 1.9, hjust = 0, colour = "grey30") +
  scale_fill_manual(values = STATUS_COLS, name = NULL, breaks = c("Case", "Control")) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = "Median disease age gap (years)", y = NULL,
       title = "Cases have a larger disease age gap than controls") +
  theme_nature +
  theme(axis.text.y = element_text(size = 6.5),
        panel.grid.major.y = element_line(colour = "grey93", linewidth = 0.2),
        legend.position = c(0.92, 0.08),
        legend.direction = "horizontal",
        legend.key.size = unit(0.3, "cm"),
        legend.text = element_text(size = 6.5))

# --- Panel d4: violin + box (raincloud-style) --------------------------------
p2d4 <- ggplot(box_all, aes(x = disease_x, y = age_gap_sex_adj, fill = status)) +
  geom_violin(width = 0.85, scale = "width", linewidth = 0.15, colour = "grey35",
              alpha = 0.55, trim = TRUE, position = position_dodge(width = 0.85)) +
  geom_boxplot(width = 0.18, outlier.shape = NA, linewidth = 0.2,
               colour = "grey20", alpha = 0.95,
               position = position_dodge(width = 0.85), show.legend = FALSE) +
  scale_fill_manual(values = STATUS_COLS, name = NULL, breaks = c("Case", "Control")) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.3) +
  coord_cartesian(ylim = c(-15, 20)) +
  scale_x_discrete(expand = expansion(add = 0.55)) +
  labs(x = NULL, y = "Disease age gap (years)",
       title = "Cases have a larger disease age gap than controls") +
  theme_nature + theme_d_wide +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))

# --- Panel d contrast variants (A/B/C/D) -------------------------------------
# Boxes are drawn from pre-computed statistics so the whiskers can be set to the
# 10-90 percentile range instead of 1.5*IQR; this removes the long outlier tails
# that compress the boxes and hide the case-control shift.
box_stats <- function(df, center_on_control = FALSE) {
  if (center_on_control) {
    ctrl_med <- df %>% filter(status == "Control") %>%
      group_by(disease_x) %>%
      summarise(ctrl_med = median(age_gap_sex_adj), .groups = "drop")
    df <- df %>% left_join(ctrl_med, by = "disease_x") %>%
      mutate(age_gap_sex_adj = age_gap_sex_adj - ctrl_med)
  }
  df %>%
    group_by(disease_x, status) %>%
    summarise(
      n      = dplyr::n(),
      ymin   = quantile(age_gap_sex_adj, 0.10),
      lower  = quantile(age_gap_sex_adj, 0.25),
      middle = median(age_gap_sex_adj),
      upper  = quantile(age_gap_sex_adj, 0.75),
      ymax   = quantile(age_gap_sex_adj, 0.90),
      .groups = "drop"
    ) %>%
    mutate(notch_half = 1.58 * (upper - lower) / sqrt(n),
           notchlower = middle - notch_half,
           notchupper = middle + notch_half)
}

delta_labels <- function(st) {
  st %>% select(disease_x, status, middle) %>%
    pivot_wider(names_from = status, values_from = middle) %>%
    mutate(delta = Case - Control,
           lab   = sprintf("%+.1f", delta))
}

make_box <- function(st, ylim, ylab, subtitle, notch = FALSE, delta = FALSE) {
  p <- ggplot(st, aes(x = disease_x, fill = status)) +
    geom_boxplot(aes(ymin = ymin, lower = lower, middle = middle,
                     upper = upper, ymax = ymax,
                     notchlower = notchlower, notchupper = notchupper),
                 stat = "identity", notch = notch,
                 width = 0.78, linewidth = 0.22,
                 position = position_dodge(width = 0.84)) +
    # drawn on top of the boxes, otherwise the zero line is hidden by them
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey25",
               linewidth = 0.3) +
    scale_fill_manual(values = STATUS_COLS, name = NULL, breaks = c("Case", "Control")) +
    scale_x_discrete(expand = expansion(add = 0.55)) +
    coord_cartesian(ylim = ylim)
  if (delta) {
    dl <- delta_labels(st)
    # delta labels occupy the top strip, so move the legend to the empty
    # bottom-left corner instead of the default top-right
    p <- p + geom_text(data = dl, aes(x = disease_x, y = ylim[2] * 0.94, label = lab),
                       inherit.aes = FALSE, size = 1.9, colour = "grey20")
  }
  p <- p +
    labs(x = NULL, y = ylab,
         title = "Cases have a larger disease age gap than controls",
         subtitle = subtitle) +
    theme_nature + theme_d_wide +
    theme(plot.subtitle = element_text(size = 6, colour = "grey35",
                                       face = "plain", margin = margin(b = 4))) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE))
  if (delta) {
    # delta labels occupy the top strip, so move the legend to the empty
    # bottom-left corner instead of the default top-right
    p <- p + theme(legend.position = c(0.01, 0.02), legend.justification = c(0, 0))
  }
  p
}

st_raw <- box_stats(box_all)
st_ctr <- box_stats(box_all, center_on_control = TRUE)

# box/whisker definitions live in the manuscript caption, not on the panel
p2dA <- make_box(st_raw, c(-8, 10), "Disease age gap (years)", NULL)
p2dB <- make_box(st_raw, c(-10, 12), "Disease age gap (years)", NULL, delta = TRUE)
p2dC <- make_box(st_raw, c(-8, 10), "Disease age gap (years)", NULL, notch = TRUE)
p2dD <- make_box(st_ctr, c(-8, 10), "Disease age gap relative to control median (years)",
                 NULL, delta = TRUE)

# --- assemble & save ---------------------------------------------------------
W_MM <- 183

save_fig <- function(plot_obj, slug, height_mm) {
  pdf_path <- file.path(FIG_DIR, paste0(slug, ".pdf"))
  png_path <- file.path(FIG_DIR, paste0(slug, ".png"))
  ggsave(pdf_path, plot_obj, width = W_MM/25.4, height = height_mm/25.4, device = "pdf")
  ggsave(png_path, plot_obj, width = W_MM/25.4, height = height_mm/25.4, dpi = 300)
  write_json(list(
    figure      = slug,
    script      = "figures/fig2/fig2.R",
    date        = format(Sys.Date()),
    sources     = c("results/disease_summary.csv", "results/bioage_scores.csv",
                    "results/sex_adjusted_gaps.csv", "data/case_control_31diseases.csv"),
    n_diseases_panel_a = N_TESTS,
    n_diseases_panel_d = nlevels(box_all$disease_x),
    calibration_r      = r_val,
    calibration_slope  = unname(slope_v),
    R_version          = R.version.string
  ), file.path(FIG_DIR, paste0(slug, ".provenance.json")), auto_unbox = TRUE, pretty = TRUE)
  cat("Saved:", pdf_path, "\n")
}

top_row <- p2a | (p2b / p2c)

# free(): release the bottom panel from patchwork's column alignment with the
# top row, so panel d can extend to the left edge of the figure instead of
# starting where panel a's plotting region starts.
assemble <- function(p_d, top = top_row) {
  (top / free(p_d, side = "l")) + plot_layout(heights = c(1, 0.48)) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 10, hjust = 0, vjust = 1),
          plot.tag.position = c(0, 1))
}

fig2_dA <- assemble(p2dA)
fig2_dB <- assemble(p2dB)
fig2_dC <- assemble(p2dC)
fig2_d2 <- assemble(p2d2)

# main deliverable: panel d = zoomed boxplot, no on-panel annotation.
# 247 mm is the Nature page limit and matches Fig 3-6; the layout is relative so
# only the vertical scale changes
save_fig(fig2_dA, "fig2",         247)

save_fig(fig2_dA, "fig2_revised_dA_zoom",       247)
save_fig(fig2_dB, "fig2_revised_dB_zoom_delta", 247)
save_fig(fig2_dC, "fig2_revised_dC_zoom_notch", 247)
save_fig(fig2_d2, "fig2_revised_d2_dumbbell",   247)

# panel-d-only previews for side-by-side comparison
save_fig(p2d1, "fig2_paneld1_box",       85)
save_fig(p2d2, "fig2_paneld2_dumbbell",  95)
save_fig(p2d4, "fig2_paneld4_violinbox", 85)

save_fig(p2b,     "fig2_panelb_dense", 90)
save_fig(p2b_hex, "fig2_panelb_hex",   90)

save_fig(p2dA, "fig2_paneldA_zoom",         85)
save_fig(p2dB, "fig2_paneldB_zoom_delta",   85)
save_fig(p2dC, "fig2_paneldC_zoom_notch",   85)
save_fig(p2dD, "fig2_paneldD_centered",     85)

cat("done\n")
