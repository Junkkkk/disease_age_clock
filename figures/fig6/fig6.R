#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Purpose : Figure 6 - Stanford ADRC external validation
#           a: AD clock (chrono age, disease age, age gap) UKB vs ADRC
#           b: PD clock (disease age, age gap)
#           c: Prevalence by age-gap threshold
#           d: LMM forest (AD age gap vs biomarkers/cognition/imaging)
#           e: Regional brain maps (amyloid PET, tau PET, cortical thickness)
# Inputs  : results/bioage_scores.csv, results/adrc/adrc_bioage_scores.csv,
#           results/adrc/adrc_lmm_results.csv,
#           results/adrc/regional_brain_associations.csv,
#           data/case_control_31diseases.csv, data/train_test_split.csv
# Outputs : output/fig6.{pdf,png} (+ provenance)
# Stage   : L5 manuscript
# Depends : src/adrc_analysis.py, src/adrc_associations.R
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(scales)
  library(cowplot); library(ggseg); library(sf)
})
stopifnot(exists("dk"))

BASE        <- normalizePath(file.path(dirname(sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
                 "..", ".."))
RES_UKB     <- file.path(BASE, "results")
RES_ADR     <- file.path(BASE, "results", "adrc")
DATA        <- file.path(BASE, "data")
FIG_DIR     <- file.path(BASE, "output")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
cat("BASE:", BASE, "\n")

# ── Data files (not distributed; set your local filenames) ───────────────────
CASE_CONTROL_FILE  <- "case_control_31diseases.csv"
SPLIT_FILE         <- "train_test_split.csv"

dpi <- 300
bs_ab <- 6.0
bs_d  <- 5.5
TAG_PT <- 10

theme_nat <- function(bs = bs_ab) {
  theme_classic(base_size = bs) +
    theme(
      text             = element_text(family = "Helvetica", size = bs),
      axis.text        = element_text(size = bs - 0.5, color = "black"),
      axis.title       = element_text(size = bs),
      plot.title       = element_text(size = bs, face = "bold", hjust = 0),
      legend.text      = element_text(size = bs - 1.5),
      legend.title     = element_text(size = bs - 1.5),
      legend.key.size  = unit(2, "mm"),
      strip.background = element_rect(fill = "grey94", color = NA),
      strip.text       = element_text(size = bs - 1, face = "bold"),
      panel.grid       = element_blank(),
      axis.line        = element_line(linewidth = 0.3),
      axis.ticks       = element_line(linewidth = 0.3),
      plot.margin      = margin(1.0, 1, 0.5, 1, "mm"),
      plot.tag         = element_text(size = TAG_PT, face = "bold", family = "Helvetica")
    )
}

inline_tag <- function(bs = bs_ab, indent = 3.2 * TAG_PT / 7) {
  theme(plot.title.position = "plot",
        plot.tag.position   = c(0, 1),
        plot.tag            = element_text(size = TAG_PT, face = "bold",
                                           family = "Helvetica", hjust = 0, vjust = 1),
        plot.title          = element_text(size = bs, face = "bold", hjust = 0,
                                           margin = margin(l = indent, b = 0.6, unit = "mm")))
}

fmt_p <- function(p) paste0("p=", formatC(p, format = "e", digits = 1))
DIAG_COLS <- c(HC="#4DAF4A", AD="#E41A1C", MCI="#FF7F00",
               PD="#984EA3", "PD-MCI"="#C77CFF", LBD="#1F78B4")

# ══════════════════════════════════════════════════════════════════════════════
# PANELS a (AD clock) + b (PD clock) — UKB + ADRC distributions
# ══════════════════════════════════════════════════════════════════════════════
cc_all   <- read.csv(file.path(DATA, CASE_CONTROL_FILE)) %>% rename(IID = eid)
ukb_all  <- read.csv(file.path(RES_UKB, "bioage_scores.csv"))
adrc_all <- read.csv(file.path(RES_ADR, "adrc_bioage_scores.csv")) %>%
  filter(diag %in% c("HC","MCI","AD","PD","PD-MCI","LBD"))

build_combined <- function(clock, ukb_col, ukb_label, main, second) {
  short  <- sub("UKB ", "", ukb_label)
  ukb_cc <- cc_all %>% select(IID, all_of(ukb_col))
  ukb <- ukb_all %>% filter(outcome == clock) %>%
    left_join(ukb_cc, by="IID") %>% filter(!is.na(.data[[ukb_col]])) %>%
    mutate(group = ifelse(.data[[ukb_col]] == 1, ukb_label, "UKB HC"),
           cohort = "UKB", bio_age = predicted_age)
  ukb$age_gap_adj <- residuals(lm(age_gap ~ chrono_age, data = ukb))
  adrc <- adrc_all %>% filter(outcome == clock) %>%
    mutate(group = paste0("ADRC ", diag), cohort = "ADRC")
  adrc$age_gap_adj <- residuals(lm(age_gap ~ chrono_age, data = adrc))

  grp_order <- c("UKB HC", ukb_label, "ADRC HC","ADRC MCI","ADRC AD",
                 "ADRC PD","ADRC PD-MCI","ADRC LBD")
  combined <- bind_rows(
    ukb  %>% select(group, cohort, chrono_age, bio_age, age_gap, age_gap_adj),
    adrc %>% select(group, cohort, chrono_age, bio_age, age_gap, age_gap_adj)
  ) %>% mutate(group = factor(group, levels = grp_order))

  main_x   <- match(paste0("ADRC ", main),   grp_order)
  second_x <- match(paste0("ADRC ", second), grp_order)
  diag_of  <- function(g) sub("ADRC |UKB ", "", g)
  gcols    <- setNames(DIAG_COLS[diag_of(grp_order)], grp_order)
  xlabs    <- setNames(c("HC", short, "HC","MCI","AD","PD","PD-MCI","LBD"), grp_order)

  stats <- combined %>% group_by(group, cohort) %>%
    summarise(n=n(), chrono_mean=mean(chrono_age), chrono_se=sd(chrono_age)/sqrt(n()),
              bio_mean=mean(bio_age), bio_se=sd(bio_age)/sqrt(n()),
              gap_mean=mean(age_gap_adj), gap_se=sd(age_gap_adj)/sqrt(n()), .groups="drop")

  pvals <- lapply(list(chrono="chrono_age", bio="bio_age", gap="age_gap_adj"), function(col)
    list(ukb       = wilcox.test(combined[[col]][combined$group=="UKB HC"],
                                 combined[[col]][combined$group==ukb_label], exact=FALSE)$p.value,
         adrc_main = wilcox.test(combined[[col]][combined$group=="ADRC HC"],
                                 combined[[col]][combined$group==paste0("ADRC ",main)], exact=FALSE)$p.value,
         adrc_second = wilcox.test(combined[[col]][combined$group=="ADRC HC"],
                                   combined[[col]][combined$group==paste0("ADRC ",second)], exact=FALSE)$p.value))

  list(combined=combined, stats=stats, pvals=pvals, gcols=gcols, xlabs=xlabs,
       main_x=main_x, second_x=second_x)
}

dot_panel <- function(data, gcols, xlabs, y_col, se_col, title, ylab,
                      main_x, second_x, hline=NULL, pvals=NULL, tag=NULL) {
  p <- ggplot(data, aes(x=group, y=.data[[y_col]], colour=group, shape=cohort)) +
    geom_errorbar(aes(ymin=.data[[y_col]]-1.96*.data[[se_col]],
                      ymax=.data[[y_col]]+1.96*.data[[se_col]]),
                  width=0.2, linewidth=0.4) +
    geom_point(size=1.5) +
    scale_colour_manual(values=gcols, guide="none") +
    scale_shape_manual(values=c(UKB=16, ADRC=17), name=NULL) +
    scale_x_discrete(labels=xlabs) +
    labs(x=NULL, y=ylab, title=title, tag=tag) +
    theme_nat() +
    theme(legend.position="none",
          axis.text.x=element_text(size=bs_ab-1, angle=45, hjust=1, vjust=1)) +
    annotate("segment", x=2.5, xend=2.5, y=-Inf, yend=Inf,
             linetype="dotted", colour="grey60", linewidth=0.25) +
    annotate("text", x=1.5, y=Inf, label="UKB",  vjust=1.6,
             size=bs_ab*0.30, colour="grey35", fontface="italic") +
    annotate("text", x=5.5, y=Inf, label="ADRC", vjust=1.6,
             size=bs_ab*0.30, colour="grey35", fontface="italic")

  if (!is.null(hline))
    p <- p + geom_hline(yintercept=hline, linetype="dashed", linewidth=0.25, colour="grey50")

  if (!is.null(pvals)) {
    yv   <- data[[y_col]] + 1.96*data[[se_col]]
    ymax <- max(yv, na.rm=TRUE)
    step <- (ymax - min(data[[y_col]]-1.96*data[[se_col]], na.rm=TRUE)) * 0.08
    tick <- step * 0.25
    bk_u <- ymax + step*0.3; bk_m <- ymax + step*1.5; bk_s <- ymax + step*2.7
    p <- p +
      annotate("segment",x=1,xend=2,y=bk_u,yend=bk_u,linewidth=0.22,colour="grey35") +
      annotate("segment",x=1,xend=1,y=bk_u-tick,yend=bk_u,linewidth=0.22,colour="grey35") +
      annotate("segment",x=2,xend=2,y=bk_u-tick,yend=bk_u,linewidth=0.22,colour="grey35") +
      annotate("text",x=1,y=bk_u+tick*0.3,label=fmt_p(pvals$ukb),size=1.8,colour="grey20",vjust=0,hjust=0) +
      annotate("segment",x=3,xend=main_x,y=bk_m,yend=bk_m,linewidth=0.22,colour="grey35") +
      annotate("segment",x=3,xend=3,y=bk_m-tick,yend=bk_m,linewidth=0.22,colour="grey35") +
      annotate("segment",x=main_x,xend=main_x,y=bk_m-tick,yend=bk_m,linewidth=0.22,colour="grey35") +
      annotate("text",x=(3+main_x)/2,y=bk_m+tick*0.3,label=fmt_p(pvals$adrc_main),size=1.8,colour="grey20",vjust=0) +
      annotate("segment",x=3,xend=second_x,y=bk_s,yend=bk_s,linewidth=0.22,colour="grey35") +
      annotate("segment",x=3,xend=3,y=bk_s-tick,yend=bk_s,linewidth=0.22,colour="grey35") +
      annotate("segment",x=second_x,xend=second_x,y=bk_s-tick,yend=bk_s,linewidth=0.22,colour="grey35") +
      annotate("text",x=(3+second_x)/2,y=bk_s+tick*0.3,label=fmt_p(pvals$adrc_second),size=1.8,colour="grey20",vjust=0)
    top <- bk_s + step*2.0
    bot <- min(data[[y_col]]-1.96*data[[se_col]], na.rm=TRUE) - step*0.3
    p   <- p + coord_cartesian(ylim=c(bot, top))
  }
  if (!is.null(tag)) p <- p + inline_tag()
  p
}

ad <- build_combined("alzheimer_disease","alzheimer_disease","UKB AD","AD","LBD")
pd <- build_combined("parkinson_disease_and_parkinsonism",
                     "parkinson_disease_and_parkinsonism","UKB PD","PD","LBD")

pA1 <- dot_panel(ad$stats, ad$gcols, ad$xlabs, "chrono_mean","chrono_se",
                 "Chronological age","Age (years)", ad$main_x, ad$second_x,
                 pvals=ad$pvals$chrono, tag="a")
pA2 <- dot_panel(ad$stats, ad$gcols, ad$xlabs, "bio_mean","bio_se",
                 "AD disease age","Age (years)", ad$main_x, ad$second_x,
                 pvals=ad$pvals$bio)
pA3 <- dot_panel(ad$stats, ad$gcols, ad$xlabs, "gap_mean","gap_se",
                 "AD age gap","Age gap (years)", ad$main_x, ad$second_x,
                 hline=0, pvals=ad$pvals$gap)
pB2 <- dot_panel(pd$stats, pd$gcols, pd$xlabs, "bio_mean","bio_se",
                 "PD disease age","Age (years)", pd$main_x, pd$second_x,
                 pvals=pd$pvals$bio, tag="b")
pB3 <- dot_panel(pd$stats, pd$gcols, pd$xlabs, "gap_mean","gap_se",
                 "PD age gap","Age gap (years)", pd$main_x, pd$second_x,
                 hline=0, pvals=pd$pvals$gap)

# ── Panel c: Prevalence by age-gap threshold ─────────────────────────────────
split_df   <- read.csv(file.path(DATA, SPLIT_FILE)) %>%
  select(IID, split) %>% distinct()
ukb_scores <- read.csv(file.path(RES_UKB,"bioage_scores.csv")) %>%
  inner_join(split_df, by="IID") %>% filter(split=="test")

cum_prev_for <- function(gap_vals, case_vals, min_n=15) {
  thresholds <- quantile(gap_vals, seq(0.05, 0.90, by=0.02))
  rows <- lapply(thresholds, function(x) {
    idx <- gap_vals >= x; n <- sum(idx); nc <- sum(case_vals[idx])
    if (n < min_n) return(NULL)
    ci <- binom.test(nc, n)$conf.int
    data.frame(threshold=x, n=n, prev=nc/n, lo=ci[1], hi=ci[2])
  })
  do.call(rbind, rows[!sapply(rows, is.null)])
}

age_match_joint <- function(ukb_df, adrc_df, bin_width=2, seed=42) {
  set.seed(seed)
  adrc_df$age_bin <- floor(adrc_df$chrono_age/bin_width)*bin_width
  ukb_df$age_bin  <- floor(ukb_df$chrono_age/bin_width)*bin_width
  do.call(rbind, lapply(c(0L,1L), function(grp) {
    rc <- table(adrc_df$age_bin[adrc_df$is_case==grp])
    ug <- ukb_df[ukb_df$is_case==grp,]
    do.call(rbind, lapply(names(rc), function(b) {
      sub <- ug[ug$age_bin==as.numeric(b),]
      if (nrow(sub)==0) return(NULL)
      sub[sample(nrow(sub), min(nrow(sub), as.integer(rc[b]))),]
    }))
  }))
}

cum_data <- bind_rows(lapply(c("alzheimer_disease","parkinson_disease_and_parkinsonism"),
  function(oc) {
    lbl <- if(oc=="alzheimer_disease") "AD" else "PD"
    clk <- if(oc=="alzheimer_disease") "AD clock" else "PD clock"
    adrc_d <- adrc_all %>% filter(outcome==oc, diag %in% c("HC",lbl)) %>%
      mutate(is_case=as.integer(diag==lbl))
    adrc_cp <- cum_prev_for(adrc_d$age_gap, adrc_d$is_case) %>%
      mutate(cohort="Stanford ADRC", clock=clk)
    ukb_f <- ukb_scores %>% filter(outcome==oc) %>%
      left_join(cc_all %>% select(IID, all_of(oc)), by="IID") %>%
      rename(is_case=all_of(oc)) %>% filter(!is.na(is_case))
    ukb_m <- age_match_joint(ukb_f, adrc_d)
    ukb_cp <- cum_prev_for(ukb_m$age_gap, ukb_m$is_case) %>%
      mutate(cohort="UKB (matched)", clock=clk)
    bind_rows(adrc_cp, ukb_cp)
  })) %>%
  mutate(clock  = factor(clock, levels=c("AD clock","PD clock")),
         cohort = factor(cohort, levels=c("Stanford ADRC","UKB (matched)")))

pATN <- ggplot(cum_data %>% filter(threshold >= -8),
               aes(x=threshold, y=prev, colour=clock, fill=clock)) +
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.15, colour=NA) +
  geom_line(linewidth=0.6) +
  scale_colour_manual(values=c("AD clock"="#E41A1C","PD clock"="#984EA3"), name=NULL) +
  scale_fill_manual(  values=c("AD clock"="#E41A1C","PD clock"="#984EA3"), name=NULL) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,NA)) +
  facet_wrap(~cohort, nrow=1, scales="free_x") +
  labs(x="Age gap threshold (years)", y="Prevalence (age gap \u2265 x)",
       tag="c", title="Prevalence by age gap threshold") +
  theme_nat() +
  theme(legend.position=c(0.13, 0.82), legend.background=element_blank(),
        legend.key.width=unit(3,"mm"),
        axis.text.x=element_text(size=bs_ab-1, angle=45, hjust=1, vjust=1),
        strip.text=element_text(size=bs_ab-1, face="bold", colour="grey20")) +
  inline_tag()

abc_grid <- plot_grid(pA1, pA2, pA3, pB2, pB3, pATN,
                      nrow=2, ncol=3, align="hv", axis="tblr")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL d — LMM forest: curated 12 outcomes, HC+MCI+AD sample only
# ══════════════════════════════════════════════════════════════════════════════
KEEP <- c("PTAU217","PTAU181","ab_ratio","GFAP","NFL",
          "b4_cdrsum","c2_mocatots","c2_craftvrs","c2_craftdvr",
          "hippo_vol_norm","amyg_vol_norm","mean_thick")
VAR_LABELS <- c(
  PTAU217="Phospho-tau 217", PTAU181="Phospho-tau 181",
  ab_ratio="Amyloid b42/40 ratio", GFAP="GFAP", NFL="Neurofilament\nlight chain",
  b4_cdrsum="CDR Sum\nof Boxes", c2_mocatots="MoCA\ntotal score",
  c2_craftvrs="Craft story recall\n(immediate)", c2_craftdvr="Craft story recall\n(delayed)",
  hippo_vol_norm="Hippocampal\nvolume (ICV-norm)", amyg_vol_norm="Amygdala\nvolume (ICV-norm)",
  mean_thick="Mean cortical\nthickness")
CAT_MAP <- c(
  PTAU217="Plasma",PTAU181="Plasma",ab_ratio="Plasma",GFAP="Plasma",NFL="Plasma",
  b4_cdrsum="Cognitive",c2_mocatots="Cognitive",c2_craftvrs="Cognitive",c2_craftdvr="Cognitive",
  hippo_vol_norm="Brain MRI",amyg_vol_norm="Brain MRI",mean_thick="Brain MRI")
CAT_COLS <- c(Plasma="#0072B2", Cognitive="#009E73", "Brain MRI"="#D55E00")

fmt_p2 <- function(p) ifelse(p < 0.001, formatC(p, format="e", digits=2), sprintf("%.3f", p))

lmm <- read.csv(file.path(RES_ADR,"adrc_lmm_results.csv"), stringsAsFactors=FALSE) %>%
  filter(group=="HC + MCI + AD", variable %in% KEEP)
var_order <- lmm %>%
  mutate(cat=factor(CAT_MAP[variable], levels=c("Plasma","Cognitive","Brain MRI"))) %>%
  arrange(cat, beta) %>% pull(variable)

plot_df <- lmm %>%
  mutate(category = factor(CAT_MAP[variable], levels=c("Plasma","Cognitive","Brain MRI")),
         var_fac  = factor(variable, levels=var_order),
         sig      = ifelse(q_val < 0.05, "*", ""),
         p_label  = paste0(fmt_p2(pval), sig))

p_lmm <- ggplot(plot_df, aes(x=beta, y=var_fac, colour=category)) +
  geom_vline(xintercept=0, linetype="dashed", colour="grey55", linewidth=0.25) +
  geom_vline(xintercept=0.043, colour="grey82", linewidth=0.25) +
  geom_pointrange(aes(xmin=lo, xmax=hi), size=0.24, linewidth=0.45) +
  geom_text(aes(label=p_label), x=0.048, hjust=0, size=1.8,
            family="Helvetica", colour="grey25") +
  scale_colour_manual(values=CAT_COLS, guide="none") +
  scale_y_discrete(labels=VAR_LABELS) +
  scale_x_continuous(breaks=c(-0.04, 0, 0.04), labels=function(x) sprintf("%.2f",x)) +
  coord_cartesian(xlim=c(-0.065, 0.105), clip="off") +
  facet_grid(category ~ ., scales="free_y", space="free_y") +
  labs(x=expression(beta~"(std., 95% CI)"), y=NULL,
       tag="d", title="AD age gap associations") +
  theme_bw(base_size=bs_d) +
  theme(
    text=element_text(family="Helvetica", size=bs_d),
    axis.text=element_text(size=bs_d, color="black"),
    axis.text.y=element_text(size=4.6, lineheight=0.80, hjust=1,
                             margin=margin(r=0.3,unit="mm")),
    axis.ticks.y=element_line(linewidth=0.2),
    axis.ticks.length.y=unit(0.8,"mm"),
    axis.title.x=element_text(size=bs_d),
    plot.title=element_text(size=bs_d+0.5, face="bold", hjust=0),
    plot.tag=element_text(size=TAG_PT, face="bold", family="Helvetica"),
    strip.background=element_rect(fill="grey92", color=NA),
    strip.text.y.right=element_text(angle=-90, hjust=0.5, vjust=0.5,
                                    size=bs_d, face="bold",
                                    margin=margin(l=0.8,r=0.8,t=0,b=0,unit="mm")),
    panel.grid.minor=element_blank(),
    panel.grid.major.x=element_line(colour="grey88", linewidth=0.2),
    panel.grid.major.y=element_line(colour="grey88", linewidth=0.12),
    panel.border=element_rect(colour="grey40"),
    plot.margin=margin(1,2,1,0,"mm")
  ) +
  inline_tag(bs = bs_d + 0.5)

g_lmm <- ggplotGrob(p_lmm)
strip_cols <- grep("strip-r", g_lmm$layout$name)
if (length(strip_cols) > 0)
  for (ci in unique(g_lmm$layout$r[strip_cols])) g_lmm$widths[ci] <- unit(3.5, "mm")
panel_d <- ggdraw() + draw_grob(g_lmm, x=0, y=0, width=1, height=1)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL e — Regional brain maps: HC+MCI+AD only, 3 modalities horizontal
# ══════════════════════════════════════════════════════════════════════════════
brain_raw <- read.csv(file.path(RES_ADR,"regional_brain_associations.csv"),
                      check.names=FALSE, stringsAsFactors=FALSE) %>%
  rename(beta=`Beta (std)`, region=Region)

dk_lh      <- dk
dk_lh$data <- dk$data[dk$data$hemi=="left",]
lat_sf <- dk_lh$data[dk_lh$data$side == "lateral", ]
med_sf <- dk_lh$data[dk_lh$data$side == "medial",  ]
lat_bb <- sf::st_bbox(lat_sf$geometry)
med_bb <- sf::st_bbox(med_sf$geometry)
lat_xmax_norm <- as.numeric(lat_bb["xmax"] - lat_bb["xmin"])
med_xmax_norm <- as.numeric(med_bb["xmax"] - med_bb["xmin"])
sep_x <- max(lat_xmax_norm, med_xmax_norm) * 1.2
brain_yrng <- max(as.numeric(lat_bb["ymax"] - lat_bb["ymin"]),
                  as.numeric(med_bb["ymax"] - med_bb["ymin"]))
suppressWarnings({
  lat_raw_cx <- sf::st_coordinates(sf::st_centroid(lat_sf$geometry))[, 1]
  lat_raw_cy <- sf::st_coordinates(sf::st_centroid(lat_sf$geometry))[, 2]
  med_raw_cx <- sf::st_coordinates(sf::st_centroid(med_sf$geometry))[, 1]
  med_raw_cy <- sf::st_coordinates(sf::st_centroid(med_sf$geometry))[, 2]
})
lat_cents <- data.frame(region = lat_sf$region, side = "lateral",
  cx = lat_raw_cx - as.numeric(lat_bb["xmin"]), cy = lat_raw_cy - as.numeric(lat_bb["ymin"]))
med_cents <- data.frame(region = med_sf$region, side = "medial",
  cx = (med_raw_cx - as.numeric(med_bb["xmin"])) + sep_x, cy = med_raw_cy - as.numeric(med_bb["ymin"]))

make_brain <- function(res_df, sample, modality, flip, title_txt, clim=NULL) {
  pd <- res_df %>% filter(Sample==sample, Modality==modality) %>%
    mutate(hemi="left", value=if(flip) -beta else beta) %>% select(hemi, region, value)
  if (nrow(pd)==0) return(ggplot()+theme_void())
  cl <- if(!is.null(clim)) clim else max(abs(pd$value),na.rm=TRUE)*1.1
  res_filt <- res_df %>% filter(Sample==sample, Modality==modality) %>% arrange(desc(abs(beta)))
  y_below <- -brain_yrng * 0.20
  lat_ok <- lat_cents[!is.na(lat_cents$region), ]
  med_ok <- med_cents[!is.na(med_cents$region), ]
  make_top <- function(cents_df, n) {
    res_filt %>% inner_join(cents_df, by="region") %>% head(n) %>%
      mutate(p_str   = ifelse(P < 0.001, formatC(P, format="e", digits=1), sprintf("%.3f", P)),
             sig_sym = ifelse(P_FDR < 0.05, "*", ""),
             ann_lab = sprintf("%s\np=%s%s", region, p_str, sig_sym))
  }
  top_lat <- make_top(lat_ok, 2); top_med <- make_top(med_ok, 2)
  lat_cx_lo <- min(lat_ok$cx, na.rm=TRUE); lat_cx_hi <- max(lat_ok$cx, na.rm=TRUE)
  med_cx_lo <- min(med_ok$cx, na.rm=TRUE); med_cx_hi <- max(med_ok$cx, na.rm=TRUE)
  place_view <- function(grp, cx_lo, cx_hi) {
    if (nrow(grp) == 0) return(grp)
    grp <- grp[order(grp$cx), ]; n <- nrow(grp); span <- cx_hi - cx_lo
    grp$lx <- if (n == 1) cx_lo + span*0.5 else seq(cx_lo + span*0.20, cx_hi - span*0.20, length.out=n)
    grp$lx <- pmax(cx_lo + span*0.05, pmin(cx_hi - span*0.05, grp$lx)); grp$ly <- y_below; grp
  }
  ann <- rbind(place_view(top_lat, lat_cx_lo, lat_cx_hi), place_view(top_med, med_cx_lo, med_cx_hi))
  p <- ggplot(pd) +
    geom_brain(atlas=dk_lh, mapping=aes(fill=value),
               position=position_brain(.~hemi+side), color="grey40", size=0.08) +
    scale_fill_distiller(palette="RdBu", direction=-1, limits=c(-cl,cl),
      na.value="grey88", name=expression(beta),
      guide=guide_colorbar(barwidth=unit(2.4,"cm"), barheight=unit(0.15,"cm"),
                           title.position="left", title.vjust=0.9)) +
    labs(title=title_txt) +
    theme_void(base_family="Helvetica") +
    theme(plot.title=element_text(size=6, face="bold", hjust=0.5, margin=margin(b=0.5,t=0.5)),
          legend.position="top", legend.title=element_text(size=4.5), legend.text=element_text(size=4),
          legend.margin=margin(0,0,0,0,"mm"), plot.background=element_rect(fill="white",color=NA),
          plot.margin=margin(0.5,0.5,8,0.5,"mm")) +
    coord_sf(clip="off", expand=FALSE)
  if (nrow(ann) > 0) {
    p <- p +
      geom_segment(data=ann, aes(x=cx, y=cy, xend=lx, yend=ly), color="grey30", linewidth=0.2,
                   arrow=arrow(length=unit(0.04,"cm"), type="closed"), inherit.aes=FALSE) +
      geom_text(data=ann, aes(x=lx, y=ly, label=ann_lab), vjust=1, size=1.7, color="grey15",
                lineheight=0.80, family="Helvetica", inherit.aes=FALSE)
  }
  p
}

SAMP <- "HC+MCI+AD"
clim_amy <- max(abs(brain_raw$beta[brain_raw$Modality=="Amyloid PET"]),       na.rm=TRUE)*1.1
clim_tau <- max(abs(brain_raw$beta[brain_raw$Modality=="Tau PET"]),            na.rm=TRUE)*1.1
clim_thk <- max(abs(brain_raw$beta[brain_raw$Modality=="Cortical thickness"]), na.rm=TRUE)*1.1
p_amy <- make_brain(brain_raw, SAMP, "Amyloid PET",        FALSE, "Amyloid PET",  clim_amy)
p_tau <- make_brain(brain_raw, SAMP, "Tau PET",            FALSE, "Tau PET",      clim_tau)
p_thk <- make_brain(brain_raw, SAMP, "Cortical thickness", TRUE,  "Cort. thickness", clim_thk)

brain_title <- ggdraw() +
  draw_label("e  Regional brain associations with AD age gap",
             x=0.005, hjust=0, size=7, fontface="bold", fontfamily="Helvetica")
brain_row  <- plot_grid(p_amy, p_tau, p_thk, nrow=1, align="hv")
panel_e    <- plot_grid(brain_title, brain_row, nrow=2, rel_heights=c(0.07, 0.93))

# ══════════════════════════════════════════════════════════════════════════════
# FINAL ASSEMBLY — top [abc | d], bottom full-width [e]
# ══════════════════════════════════════════════════════════════════════════════
W_total_mm <- 183
W_right_mm <- 60
W_left_mm  <- W_total_mm - W_right_mm
H_top_mm   <- 95
H_e_mm     <- 48
H_total_mm <- H_top_mm + H_e_mm

top_row  <- plot_grid(abc_grid, panel_d, nrow=1, rel_widths=c(W_left_mm, W_right_mm))
combined <- plot_grid(top_row, panel_e, nrow=2, rel_heights=c(H_top_mm, H_e_mm))

stem <- file.path(FIG_DIR, "fig6")
for (ext in c("png","pdf")) {
  fn <- paste0(stem, ".", ext)
  if (ext=="png") png(fn, width=W_total_mm, height=H_total_mm, units="mm", res=dpi)
  else            pdf(fn, width=W_total_mm/25.4, height=H_total_mm/25.4)
  print(combined); dev.off()
  cat("Saved:", fn, "\n")
}

prov <- sprintf('{
  "figure": "fig6",
  "script": "figures/fig6/fig6.R",
  "date": "%s",
  "inputs": ["results/adrc/adrc_bioage_scores.csv", "results/bioage_scores.csv",
             "data/case_control_31diseases.csv", "data/train_test_split.csv",
             "results/adrc/adrc_lmm_results.csv",
             "results/adrc/regional_brain_associations.csv"],
  "panels": {
    "a": "Chronological age | AD disease age | AD age gap, UKB vs ADRC groups",
    "b": "PD disease age | PD age gap",
    "c": "Prevalence by age-gap threshold (Stanford ADRC vs UKB matched)",
    "d": "LMM forest, curated 12 outcomes, HC+MCI+AD sample only",
    "e": "Regional brain maps (HC+MCI+AD), 3 modalities horizontal"
  },
  "R_version": "%s"
}', Sys.Date(), R.version.string)
writeLines(prov, paste0(stem, ".provenance.json"))
cat("Done:", round(W_total_mm,1), "x", round(H_total_mm,1), "mm\n")
