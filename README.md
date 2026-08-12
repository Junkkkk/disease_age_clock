# Disease-Specific Proteomic Aging Clocks
**"Disease-specific proteomic aging clocks as maximized mediators of disease incidence"**

## Overview

This repository contains the analysis code for constructing disease-specific
biological aging clocks from plasma proteomics, optimised as maximum mediators
(MaxIE) of disease incidence. The clocks are trained on UK Biobank Olink
Explore 3k data (n = 44,526) across 31 age-related diseases and externally
validated in the Stanford ADRC cohort (n = 501).

Raw data are not included due to data use agreements (UK Biobank, Stanford
ADRC).

## Repository structure

```
src/
  optmed.py               # Core MaxIE algorithm: closed-form solver,
                          #   cosine test, mediation statistics
  ukb_analysis.py         # UKB discovery pipeline: MaxIE fitting,
                          #   calibration, Cox HR, KM, onset timing,
                          #   age regression, R² decomposition
  enrichment.R            # Pathway enrichment (fgsea) on clock weights
  adrc_analysis.py        # ADRC validation: protein harmonization,
                          #   intercept batch correction, model application
  adrc_associations.R     # ADRC downstream: LMM associations,
                          #   regional brain surface analysis
figures/
  fig2/fig2.R             # Cosine test, calibration, age gap distribution
  fig3/fig3.R             # Disease clock vs organ clock comparison
  fig4/fig4.R             # OR, HR, KM curves, age gap vs onset
  fig5/fig5.R             # Weight correlation, UMAP, pathway enrichment
  fig6/fig6.R             # ADRC external validation
```

## Pipeline

### UKB discovery (Figures 2–5)

1. **MaxIE clock construction** (`src/ukb_analysis.py`, Section 1)
   — For each of 31 diseases, the cosine test (`cosine_test()` from
   `optmed.py`) evaluates whether any linear protein composite mediates the
   age-to-disease path. For the 20 diseases passing significance,
   `solve_maxie()` returns the closed-form weight vector maximizing the
   Baron–Kenny indirect effect. Scores are calibrated to year-equivalent
   biological age gaps via sex-personalized OLS regression. All weights,
   normalization parameters, and calibration coefficients are estimated on
   the training set and frozen.

2. **Downstream analyses** (`src/ukb_analysis.py`, Sections 2–4)
   — Cox proportional hazards (per year of age gap, adjusting for
   chronological age), Kaplan–Meier cumulative incidence (10th/90th
   percentile stratification), time-to-onset regression, conventional age
   regression weights, and sequential R² variance decomposition.

3. **Pathway enrichment** (`src/enrichment.R`)
   — Pre-ranked fgseaMultilevel on |w| against Hallmark, KEGG, Reactome,
   and GO:BP (MSigDB via msigdbr). BH-corrected within each database.

### ADRC external validation (Figure 6)

4. **Model application** (`src/adrc_analysis.py`)
   — Olink protein IDs are mapped to gene symbols. Per-protein
   intercept batch correction removes platform mean shifts between ADRC and
   UKB (age-regression intercept difference, estimated once from ADRC
   healthy controls). Frozen UKB weights and calibration coefficients are
   applied without modification.

5. **Association analyses** (`src/adrc_associations.R`)
   — Linear mixed models (age gap ~ plasma biomarkers / cognitive scores /
   ICV-normalised brain volumes, with random intercepts per subject) and
   region-wise brain surface analysis (amyloid PET, tau PET, cortical
   thickness on the Desikan–Killiany atlas).

### Figure generation

6. **Figures 2–6** (`figures/fig{2,3,4,5,6}/`)
   — Each R script reads aggregate pipeline outputs (`results/*.csv`) and
   produces the corresponding manuscript figure.

## Dependencies

**Python** (≥ 3.11): numpy, scipy, pandas, scikit-learn, lifelines

**R** (≥ 4.3): ggplot2, dplyr, tidyr, readr, patchwork, ggrepel, scales,
forcats, hexbin, stringr, uwot, fgsea, msigdbr, lme4, cowplot, ggseg, sf,
jsonlite

## Contact

Junyoung Park (jpark01@stanford.edu)
