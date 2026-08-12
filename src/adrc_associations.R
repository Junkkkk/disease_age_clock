#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : ADRC downstream associations — LMM (age gap vs biomarkers/cognition/
#           imaging) and regional brain surface analysis (amyloid PET, tau PET,
#           cortical thickness)
# Inputs  : results/adrc/lmm_analysis_data.csv,
#           results/adrc/adrc_longitudinal_scores.csv,
#           ADRC diagnosis + imaging files (not distributed)
# Outputs : results/adrc/adrc_lmm_results.csv,
#           results/adrc/regional_brain_associations.csv
# Stage   : L4 analysis (ADRC validation, feeds Figure 6 panels d-e)
# Depends : src/adrc_analysis.py (produces lmm_analysis_data, longitudinal_scores)
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(lme4)
})

BASE        <- normalizePath(file.path(dirname(sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
                 ".."))
RES_ADR     <- file.path(BASE, "results", "adrc")

# ── ADRC data directory (not distributed; set your local path) ───────────────
ADRC_DATA_DIR <- "path/to/adrc/data"
cat("BASE:", BASE, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LMM analysis — AD age gap vs biomarkers / cognition / imaging
# ══════════════════════════════════════════════════════════════════════════════
cat("Running LMM analysis ...\n")

dat_lmm <- read.csv(file.path(RES_ADR, "lmm_analysis_data.csv"),
                    na.strings = c("", "NA"), stringsAsFactors = FALSE)
dat_lmm$sex <- factor(dat_lmm$sex, levels = c("F", "M"))

# Variable definitions
biomarker_vars <- c("PTAU217", "PTAU181", "ab_ratio", "GFAP", "NFL")
cog_vars       <- c("c2_mocatots", "c2_craftvrs", "c2_craftdvr", "c2_animals",
                    "c2_digforct", "c2_digbacct", "c2_traila", "c2_trailb",
                    "b4_cdrsum")
struct_vars    <- c("hippo_vol_norm", "amyg_vol_norm", "vent_vol_norm",
                    "wmh_norm", "brain_vol_norm", "mean_thick")
all_vars_lmm   <- c(biomarker_vars, cog_vars, struct_vars)

groups_lmm <- list(
  "HC"            = c("HC"),
  "HC + MCI"      = c("HC", "MCI"),
  "HC + MCI + AD" = c("HC", "MCI", "AD")
)

run_lmm <- function(outcome_var, sub_df, min_n = 20, min_subj = 10) {
  d <- sub_df[, c("adrc_id", "age_gap", "chrono_age", "sex", outcome_var)]
  d <- d[complete.cases(d), ]
  d <- d[is.finite(d[[outcome_var]]), ]
  if (nrow(d) < min_n) return(NULL)
  n_subj <- length(unique(d$adrc_id))
  if (n_subj < min_subj) return(NULL)
  d[[outcome_var]] <- scale(d[[outcome_var]])[, 1]
  use_lmm    <- n_subj < nrow(d)
  model_type <- if (use_lmm) "LMM" else "OLS"
  if (use_lmm) {
    fml <- as.formula(paste0("`", outcome_var,
                             "` ~ age_gap + chrono_age + sex + (1 | adrc_id)"))
    fit <- tryCatch(
      lmer(fml, data = d, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
      error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    cf <- summary(fit)$coefficients
  } else {
    fml <- as.formula(paste0("`", outcome_var, "` ~ age_gap + chrono_age + sex"))
    fit <- tryCatch(lm(fml, data = d), error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    cf <- summary(fit)$coefficients
  }
  if (!"age_gap" %in% rownames(cf)) return(NULL)
  data.frame(
    beta       = cf["age_gap", "Estimate"],
    se         = cf["age_gap", "Std. Error"],
    pval       = cf["age_gap", "Pr(>|t|)"],
    n_obs      = nrow(d),
    n_subj     = n_subj,
    model_type = model_type,
    stringsAsFactors = FALSE
  )
}

results_list_lmm <- list()
for (grp_name in names(groups_lmm)) {
  diags_lmm <- groups_lmm[[grp_name]]
  sub_df    <- dat_lmm[dat_lmm$diag %in% diags_lmm, ]
  for (v in all_vars_lmm) {
    res <- run_lmm(v, sub_df)
    if (!is.null(res)) {
      res$variable <- v
      res$group    <- grp_name
      results_list_lmm[[paste(grp_name, v, sep = "||")]] <- res
    }
  }
}

results_df_lmm <- do.call(rbind, results_list_lmm)
rownames(results_df_lmm) <- NULL
results_df_lmm <- results_df_lmm %>%
  group_by(group) %>%
  mutate(q_val = p.adjust(pval, method = "BH"),
         sig   = case_when(q_val < 0.001 ~ "***", q_val < 0.01 ~ "**",
                           q_val < 0.05  ~ "*",   TRUE ~ "")) %>%
  ungroup() %>%
  mutate(lo = beta - 1.96 * se,
         hi = beta + 1.96 * se)

write.csv(results_df_lmm, file.path(RES_ADR, "adrc_lmm_results.csv"),
          row.names = FALSE)
cat("Saved LMM results:", nrow(results_df_lmm), "tests\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Brain surface analysis — AD age gap vs regional PET/thickness
# ══════════════════════════════════════════════════════════════════════════════
cat("Running brain surface analysis ...\n")

LONG_FILE <- file.path(RES_ADR, "adrc_longitudinal_scores.csv")

# FreeSurfer → ggseg region name mapping
fs_to_ggseg <- c(
  bankssts="bankssts", caudalanteriorcingulate="caudal anterior cingulate",
  caudalmiddlefrontal="caudal middle frontal", cuneus="cuneus",
  entorhinal="entorhinal", frontalpole="frontal pole", fusiform="fusiform",
  inferiorparietal="inferior parietal", inferiortemporal="inferior temporal",
  insula="insula", isthmuscingulate="isthmus cingulate",
  lateraloccipital="lateral occipital", lateralorbitofrontal="lateral orbitofrontal",
  lingual="lingual", medialorbitofrontal="medial orbitofrontal",
  middletemporal="middle temporal", paracentral="paracentral",
  parahippocampal="parahippocampal", parsopercularis="pars opercularis",
  parsorbitalis="pars orbitalis", parstriangularis="pars triangularis",
  pericalcarine="pericalcarine", postcentral="postcentral",
  posteriorcingulate="posterior cingulate", precentral="precentral",
  precuneus="precuneus", rostralanteriorcingulate="rostral anterior cingulate",
  rostralmiddlefrontal="rostral middle frontal", superiorfrontal="superior frontal",
  superiorparietal="superior parietal", superiortemporal="superior temporal",
  supramarginal="supramarginal", temporalpole="temporal pole",
  transversetemporal="transverse temporal"
)

# Load Olink age_gap scores
scores_brain <- read.csv(LONG_FILE, stringsAsFactors = FALSE) %>%
  filter(outcome == "alzheimer_disease") %>%
  mutate(adrc_id    = suppressWarnings(as.integer(adrc_id)),
         chrono_age = as.numeric(chrono_age),
         age_gap    = as.numeric(age_gap)) %>%
  filter(!is.na(adrc_id), !is.na(chrono_age), !is.na(age_gap)) %>%
  select(adrc_id, visit, age_gap, chrono_age, sex) %>%
  group_by(adrc_id, visit) %>% slice(1) %>% ungroup()

cat(sprintf("Olink scores: %d rows, %d subjects\n",
            nrow(scores_brain), length(unique(scores_brain$adrc_id))))

# Load diagnosis, match Olink → diagnosis by closest age
diag_brain <- read.csv(file.path(ADRC_DATA_DIR, "diagnosis.csv"),
                       check.names = FALSE) %>%
  mutate(adrc_id         = suppressWarnings(as.integer(adrc_id)),
         age             = suppressWarnings(as.numeric(age)),
         diagnosis_visit = suppressWarnings(as.integer(diagnosis_visit))) %>%
  filter(!is.na(adrc_id), !is.na(age), !is.na(diagnosis_visit)) %>%
  select(adrc_id, diag_age = age, diagnosis_visit, diagnosis_consensus)

olink_matched <- scores_brain %>%
  inner_join(diag_brain, by = "adrc_id", relationship = "many-to-many") %>%
  mutate(age_diff = abs(chrono_age - diag_age)) %>%
  group_by(adrc_id, visit) %>%
  slice_min(age_diff, n = 1, with_ties = FALSE) %>%
  ungroup()

# Restrict to AD-relevant diagnoses
olink_matched <- olink_matched %>%
  filter(!grepl("Parkinson|Lewy|Vascular|FTD|other|Other",
                diagnosis_consensus, ignore.case = TRUE))

olink_hcmci <- olink_matched %>%
  filter(grepl("Healthy Control|Mild Cognitive Impairment|\\bHC\\b|\\bMCI\\b",
               diagnosis_consensus, ignore.case = TRUE))
olink_all   <- olink_matched

# Load imaging files
load_imaging <- function(fname) {
  read.csv(file.path(ADRC_DATA_DIR, fname), check.names = FALSE,
           stringsAsFactors = FALSE) %>%
    mutate(adrc_id         = suppressWarnings(as.integer(adrc_id)),
           diagnosis_visit = suppressWarnings(as.integer(diagnosis_visit))) %>%
    filter(!is.na(adrc_id), !is.na(diagnosis_visit))
}

# Imaging phenotype files (not distributed)
AMYLOID_FILE   <- "imaging_phenotypes_amyloid_thickness.csv"
TAU_THICK_FILE <- "imaging_phenotypes_tau_thickness.csv"

amy_raw <- load_imaging(AMYLOID_FILE)
tau_raw <- load_imaging(TAU_THICK_FILE)

merge_imaging <- function(img_df, olink_df) {
  inner_join(
    img_df  %>% rename(img_visit = diagnosis_visit),
    olink_df %>% select(adrc_id, olink_visit = diagnosis_visit,
                        age_gap, chrono_age, sex, age_diff, diagnosis_consensus),
    by = "adrc_id", relationship = "many-to-many"
  ) %>%
    mutate(visit_diff = abs(img_visit - olink_visit)) %>%
    group_by(adrc_id, img_visit) %>%
    slice_min(visit_diff, n = 1, with_ties = FALSE) %>%
    ungroup()
}

amy_hcmci <- merge_imaging(amy_raw, olink_hcmci)
tau_hcmci <- merge_imaging(tau_raw, olink_hcmci)
amy_all   <- merge_imaging(amy_raw, olink_all)
tau_all   <- merge_imaging(tau_raw, olink_all)

# Run LMM per region
run_assoc <- function(merged_df, lh_pat, prefix_strip,
                      suffix_strip = NULL, modality, min_n = 15) {
  all_cols <- colnames(merged_df)
  lh_cols  <- grep(lh_pat, all_cols, value = TRUE)
  fs_names <- gsub(prefix_strip, "", lh_cols)
  if (!is.null(suffix_strip)) fs_names <- gsub(suffix_strip, "", fs_names)

  results_brain <- lapply(seq_along(lh_cols), function(i) {
    fs_name <- fs_names[i]
    gname   <- fs_to_ggseg[fs_name]
    if (is.na(gname)) return(NULL)
    lh_col <- lh_cols[i]
    rh_col <- sub("ctx-lh-", "ctx-rh-", lh_col)
    rh_col <- sub("^lh_",    "rh_",     rh_col)
    if (!rh_col %in% all_cols) rh_col <- NULL
    lh_vals <- suppressWarnings(as.numeric(merged_df[[lh_col]]))
    rh_vals <- if (!is.null(rh_col))
                 suppressWarnings(as.numeric(merged_df[[rh_col]])) else NA
    bilat <- rowMeans(cbind(lh_vals, rh_vals), na.rm = TRUE)
    bilat[is.nan(bilat)] <- NA
    d <- data.frame(
      y       = bilat,
      age_gap = merged_df$age_gap,
      age     = merged_df$chrono_age,
      sex     = factor(merged_df$sex),
      subject = factor(merged_df$adrc_id)
    ) %>% filter(!is.na(y), !is.na(age_gap), !is.na(age))
    if (nrow(d) < min_n) return(NULL)
    d$y_z       <- scale(d$y)[, 1]
    d$age_gap_z <- scale(d$age_gap)[, 1]
    d$age_z     <- scale(d$age)[, 1]
    n.subj_b    <- length(unique(d$subject))
    use_lmm_b   <- n.subj_b < nrow(d)
    model_type_b <- if (use_lmm_b) "LMM" else "OLS"
    ct <- tryCatch({
      if (use_lmm_b) {
        fit <- lmer(y_z ~ age_gap_z + age_z + sex + (1 | subject), data = d,
                    REML = FALSE,
                    control = lmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 2e5)))
        as.data.frame(coef(summary(fit)))
      } else {
        as.data.frame(summary(lm(y_z ~ age_gap_z + age_z + sex, data = d))$coefficients)
      }
    }, error = function(e) NULL)
    if (is.null(ct) || !"age_gap_z" %in% rownames(ct)) return(NULL)
    rownames(ct)[rownames(ct) == "age_gap_z"] <- "age_gap"
    p_col <- grep("Pr\\(", colnames(ct), value = TRUE)[1]
    t_col <- grep("^t |^t$|t value", colnames(ct), value = TRUE)[1]
    data.frame(modality = modality, region = gname, model = model_type_b,
               n.obs = nrow(d), n.subj = n.subj_b,
               beta  = ct[["age_gap", "Estimate"]],
               se    = ct[["age_gap", "Std. Error"]],
               t_val = ct[[t_col]][rownames(ct) == "age_gap"],
               p_val = ct[[p_col]][rownames(ct) == "age_gap"],
               stringsAsFactors = FALSE)
  })
  res_b <- bind_rows(Filter(Negate(is.null), results_brain))
  if (nrow(res_b) > 0) res_b$p_fdr <- p.adjust(res_b$p_val, method = "BH")
  res_b
}

run_all_mods <- function(amy_df, tau_df, sample) {
  amy   <- run_assoc(amy_df, "Mean[.]ctx-lh-", "^Mean[.]ctx-(lh|rh)-",
                     modality = "Amyloid PET")
  tau   <- run_assoc(tau_df, "Mean[.]ctx-lh-", "^Mean[.]ctx-(lh|rh)-",
                     modality = "Tau PET")
  thick <- run_assoc(tau_df, "^lh_", "^(lh|rh)_", "_thickness$",
                     modality = "Cortical thickness")
  all_b <- bind_rows(amy, tau, thick)
  cat(sprintf("\n[%s] Total: %d tests | FDR<0.05: %d | Nominal p<0.05: %d\n",
              sample, nrow(all_b),
              sum(all_b$p_fdr < 0.05, na.rm=TRUE),
              sum(all_b$p_val < 0.05, na.rm=TRUE)))
  list(amy = amy, tau = tau, thick = thick, all = all_b, sample = sample)
}

res_hcmci <- run_all_mods(amy_hcmci, tau_hcmci, "HC+MCI")
res_all   <- run_all_mods(amy_all,   tau_all,   "HC+MCI+AD")

# Save combined results table
format_table <- function(res_list) {
  res_list$all %>%
    mutate(
      sample = res_list$sample,
      sig    = case_when(p_fdr < 0.001 ~ "***", p_fdr < 0.01 ~ "**",
                         p_fdr < 0.05  ~ "*",   p_val < 0.05 ~ "\u2020",
                         TRUE ~ "")
    ) %>%
    arrange(modality, p_val) %>%
    transmute(
      Sample       = sample,
      Modality     = modality,
      Region       = region,
      Model        = model,
      N_obs        = n.obs,
      N_subjects   = n.subj,
      `Beta (std)` = round(beta,  4),
      SE           = round(se,    4),
      t            = round(t_val, 3),
      P            = signif(p_val, 3),
      P_FDR        = signif(p_fdr, 3),
      `Sig.`       = sig
    )
}

table_out <- bind_rows(format_table(res_hcmci), format_table(res_all))
write.csv(table_out, file.path(RES_ADR, "regional_brain_associations.csv"),
          row.names = FALSE)
cat(sprintf("Saved: %s/regional_brain_associations.csv (%d rows)\n",
            RES_ADR, nrow(table_out)))
