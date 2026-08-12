"""
Purpose : Apply UKB-trained disease aging clocks to Stanford ADRC proteomics
Inputs  : results/all_weights.csv, results/normalization_params.csv,
          results/calibration_params.csv, ADRC plasma proteomics + demographics
Outputs : results/adrc/adrc_bioage_scores.csv, results/adrc/lmm_analysis_data.csv,
          results/adrc/adrc_longitudinal_scores.csv
Stage   : L4 analysis (external validation)
Depends : src/ukb_analysis.py (produces weights, normalization, calibration,
          age_regression_weights)

Clock application:
  MaxIE_score   = sum(w_k * z_k)   where z_k = (x_k - mu_k) / sd_k
  age_gap       = (MaxIE_score - b0 - chrono_age * b1 - sex * b2) / b1
  predicted_age = chrono_age + age_gap
"""

import os
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

# ── Paths (not distributed; set your local paths) ────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = "path/to/adrc/data"            # ADRC data directory
RES_UKB    = os.path.join(BASE, "results")
RES_ADR    = os.path.join(BASE, "results", "adrc")
os.makedirs(RES_ADR, exist_ok=True)

# ADRC data files (not distributed)
PROTEIN_INFO_FILE  = "protein_info_plasma.csv"   # ProteinId → gene symbol mapping
PROTEOMICS_FILE    = "proteomics_plasma.csv"      # Olink plasma proteomics
DEMOGRAPHICS_FILE  = "demographics.csv"           # adrc_id, sex
DIAGNOSIS_FILE     = "diagnosis.csv"              # adrc_id, diagnosis_consensus
BIOMARKERS_FILE    = "biomarkers.csv"             # pTau, GFAP, NFL etc.
COGNITIVE_FILE     = "cognitive_scores.csv"        # MoCA, Craft, CDR etc.
IMAGING_FILE       = "imaging_phenotypes_tau_thickness.csv"  # FreeSurfer volumes

# ── Diagnosis consensus → short label ────────────────────────────────────────
def map_diagnosis(d):
    d = str(d).strip().split(";")[0].strip()
    if "Healthy Control" in d:
        return "HC"
    if "Probable Alzheimer" in d or "Possible Alzheimer" in d:
        return "AD"
    if "Mild Cognitive Impairment" in d:
        return "MCI"
    if "Lewy Body Disease" in d:
        return "LBD"
    if "Parkinson" in d and "Mild cognitive" in d:
        return "PD-MCI"
    if "Parkinson" in d:
        return "PD"
    return "Other"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load UKB reference data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading UKB reference files ...")
weights     = pd.read_csv(os.path.join(RES_UKB, "all_weights.csv"))
norm_params = pd.read_csv(os.path.join(RES_UKB, "normalization_params.csv"))
cal_params  = pd.read_csv(os.path.join(RES_UKB, "calibration_params.csv"))

norm_params = norm_params.set_index("protein")
disease_cols = [c for c in weights.columns if c != "protein"]
weights = weights.set_index("protein")

print(f"  {len(disease_cols)} disease clocks, {len(weights)} proteins")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load and harmonize ADRC proteomics
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading ADRC data ...")

# Protein ID → gene symbol (lowercase) mapping
prot_info = pd.read_csv(os.path.join(DATA_DIR, PROTEIN_INFO_FILE))
pid_to_gene = dict(zip(prot_info["ProteinId"],
                       prot_info["EntrezGeneSymbol"].str.lower()))

# Plasma proteomics
plasma = pd.read_csv(os.path.join(DATA_DIR, PROTEOMICS_FILE))
meta_cols = ["adrc_id", "pidn", "year_quarter", "diagnosis_visit", "date_of_birth"]
prot_cols  = [c for c in plasma.columns if c not in meta_cols and c in pid_to_gene]

# Rename protein columns to gene symbols (lowercase, matching UKB convention)
plasma_prot = plasma[meta_cols + prot_cols].copy()
rename_map  = {c: pid_to_gene[c] for c in prot_cols}
plasma_prot.rename(columns=rename_map, inplace=True)

# Multiple aptamers can map to the same gene → deduplicate by mean
gene_cols_raw = list(rename_map.values())
dup_genes = [g for g in set(gene_cols_raw) if gene_cols_raw.count(g) > 1]
if dup_genes:
    meta_df = plasma_prot[meta_cols].copy()
    prot_df = plasma_prot[gene_cols_raw]
    prot_df = prot_df.T.groupby(level=0).mean().T
    plasma_prot = pd.concat([meta_df, prot_df], axis=1)
gene_cols = [c for c in plasma_prot.columns if c not in meta_cols]

# Demographics (sex)
demo = pd.read_csv(os.path.join(DATA_DIR, DEMOGRAPHICS_FILE))
demo["sex_num"] = (demo["sex"].str.lower() == "male").astype(float)
demo["sex_str"] = demo["sex"].str.lower().map({"male": "M", "female": "F"})

# Diagnosis
diag_df = pd.read_csv(os.path.join(DATA_DIR, DIAGNOSIS_FILE))
diag_df["diag"] = diag_df["diagnosis_consensus"].apply(map_diagnosis)

print(f"  plasma: {plasma.shape[0]} rows, {len(gene_cols)} protein cols mapped")
print(f"  diagnosis: {diag_df['diag'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Match proteins to UKB weight space
# ═══════════════════════════════════════════════════════════════════════════════
ukb_proteins = set(weights.index) & set(norm_params.index)
common_genes  = sorted(ukb_proteins & set(gene_cols))
print(f"  ADRC/UKB overlap: {len(common_genes)} proteins (non-overlapping imputed to 0)")

W  = weights.loc[common_genes, disease_cols].values          # (P, D)
mu = norm_params.loc[common_genes, "mu"].values               # (P,)
sd = norm_params.loc[common_genes, "sd"].values               # (P,)
sd = np.where(sd < 1e-10, 1.0, sd)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Intercept batch correction
#
#    For each protein j, fit X_j = a0_j + a1_j * Age independently in each
#    cohort.  The platform batch offset is the intercept difference:
#
#        batch_j = a0_j^ADRC - a0_j^UKB
#
#    Corrected z-score for any individual i:
#
#        z_j(i) = [x_j(i) - batch_j - mu_j^UKB] / sd_j^UKB
#
#    This removes the age-independent platform mean shift while preserving
#    the age-linear protein signal.  batch_j is a frozen per-protein constant
#    valid for any individual from the same ADRC platform.
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing intercept batch correction ...")

X_raw = plasma_prot[common_genes].values.astype(float)       # (N, P)
X_raw = np.where(np.isnan(X_raw), mu[None, :], X_raw)        # impute missing → UKB mean

# Identify HC subjects for fitting ADRC intercepts
diag_lookup_bc = diag_df.groupby(["adrc_id", "diagnosis_visit"])["diag"].first().reset_index()
base_bc = plasma_prot[["adrc_id", "diagnosis_visit"]].copy()
base_bc = base_bc.merge(diag_lookup_bc, on=["adrc_id", "diagnosis_visit"], how="left")
base_bc = base_bc.merge(demo[["adrc_id", "sex_num"]], on="adrc_id", how="left")

# Compute age for batch correction (same logic as below)
plasma_prot_bc = plasma_prot.copy()
plasma_prot_bc["visit_year"] = plasma_prot_bc["year_quarter"].apply(
    lambda yq: float(yq.split("_")[0]) + (int(yq.split("_")[1]) - 1) * 0.25 + 0.125
    if isinstance(yq, str) and "_" in str(yq) else np.nan)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dob_bc = pd.to_datetime(plasma[["date_of_birth"]].iloc[:len(plasma_prot_bc)]["date_of_birth"], errors="coerce")
    birth_yr_bc = dob_bc.dt.year + (dob_bc.dt.month - 1) / 12.0 + dob_bc.dt.day / 365.25
base_bc["chrono_age"] = plasma_prot_bc["visit_year"].values - birth_yr_bc.values

# Fallback age from diagnosis
diag_age_bc = diag_df[["adrc_id", "age", "diagnosis_visit"]].copy()
base_bc = base_bc.merge(diag_age_bc.rename(columns={"age": "diag_age"}),
                        on=["adrc_id", "diagnosis_visit"], how="left")
base_bc.loc[base_bc["chrono_age"].isna() | (base_bc["chrono_age"] <= 0),
            "chrono_age"] = base_bc["diag_age"]

hc_mask = (base_bc["diag"] == "HC") & base_bc["chrono_age"].notna()
A_hc = base_bc.loc[hc_mask, "chrono_age"].values
X_hc = X_raw[hc_mask.values]
print(f"  ADRC HC subjects for batch correction: {hc_mask.sum()}")

# Per-protein OLS intercept in ADRC HC: X_j = a0_j + a1_j * Age
design_hc = np.column_stack([np.ones(len(A_hc)), A_hc])
coefs_adrc = lstsq(design_hc, X_hc, rcond=None)[0]          # (2, P)
a0_adrc = coefs_adrc[0]                                       # (P,)

# Per-protein OLS intercept in UKB training:
#   a1_j^UKB = r_j * sd_j / sd_A,  a0_j^UKB = mu_j - a1_j * mean_A
age_reg = pd.read_csv(os.path.join(RES_UKB, "age_regression_weights.csv"))
age_reg = age_reg.set_index("protein").reindex(common_genes)
r_age = age_reg["weight"].fillna(0).values

UKB_MEAN_AGE = 56.54    # UKB training mean age (years)
UKB_SD_AGE   = 8.27     # UKB training SD age (years)
a1_ukb = r_age * sd / UKB_SD_AGE
a0_ukb = mu - a1_ukb * UKB_MEAN_AGE

# Batch offset: per-protein platform intercept difference
batch = a0_adrc - a0_ukb                                      # (P,)
print(f"  Batch offset mean={batch.mean():.4f}  |batch| mean={np.abs(batch).mean():.4f}")

# Corrected z-scores: z_j = (x_j - batch_j - mu_j) / sd_j
Z = (X_raw - batch[None, :] - mu[None, :]) / sd[None, :]     # (N, P)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Compute MaxIE scores for each visit
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing MaxIE scores ...")

# MaxIE scores: (N, D)
scores = Z @ W

# Assemble base dataframe
base = plasma_prot[["adrc_id", "diagnosis_visit", "year_quarter"]].copy()
base = base.merge(demo[["adrc_id", "sex_num", "sex_str"]], on="adrc_id", how="left")

# Age at visit from date_of_birth + year_quarter
def yq_to_decimal(yq):
    try:
        yr, q = yq.split("_")
        return float(yr) + (int(q) - 1) * 0.25 + 0.125
    except Exception:
        return np.nan

plasma_prot["visit_year"] = plasma_prot["year_quarter"].apply(yq_to_decimal)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dob = pd.to_datetime(plasma["date_of_birth"], errors="coerce")
    birth_yr = dob.dt.year + (dob.dt.month - 1) / 12.0 + dob.dt.day / 365.25

base["birth_yr"]  = birth_yr.values
base["visit_year"] = plasma_prot["visit_year"].values
base["chrono_age"] = base["visit_year"] - base["birth_yr"]

# Fallback: use age from diagnosis.csv when DOB is missing
diag_age = diag_df[["adrc_id", "age", "diagnosis_visit"]].copy()
base = base.merge(diag_age.rename(columns={"age": "diag_age"}),
                  on=["adrc_id", "diagnosis_visit"], how="left")
base.loc[base["chrono_age"].isna() | (base["chrono_age"] <= 0),
         "chrono_age"] = base["diag_age"]
base.drop(columns=["diag_age", "birth_yr", "visit_year"], inplace=True)

base = base[base["chrono_age"].notna() & (base["chrono_age"] > 0)].copy()
scores_filt = scores[base.index, :]
base.reset_index(drop=True, inplace=True)

# Apply calibration: age_gap = (M - b0 - chrono_age*b1 - sex*b2) / b1
cal = cal_params.set_index("outcome")
rows = []
for i, disease in enumerate(disease_cols):
    b0 = cal.loc[disease, "b0"]
    b1 = cal.loc[disease, "b1"]
    b2 = cal.loc[disease, "b2"]
    M  = scores_filt[:, i]
    chrono = base["chrono_age"].values
    sex    = base["sex_num"].fillna(0).values
    age_gap      = (M - b0 - chrono * b1 - sex * b2) / (b1 + 1e-15)
    predicted_age = chrono + age_gap
    tmp = base[["adrc_id", "diagnosis_visit", "year_quarter", "sex_str",
                "sex_num", "chrono_age"]].copy()
    tmp["outcome"]        = disease
    tmp["MaxIE_score"]    = M
    tmp["age_gap"]        = age_gap
    tmp["predicted_age"]  = predicted_age
    rows.append(tmp)

bio_scores = pd.concat(rows, ignore_index=True)

# Attach diagnosis label
diag_lookup = diag_df.groupby(["adrc_id", "diagnosis_visit"])[
    ["diag", "diagnosis_consensus"]].first().reset_index()
bio_scores  = bio_scores.merge(diag_lookup, on=["adrc_id", "diagnosis_visit"], how="left")
bio_scores["diag"].fillna("Other", inplace=True)

print(f"  bio_scores shape: {bio_scores.shape}  |  "
      f"subjects: {bio_scores['adrc_id'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save adrc_bioage_scores.csv
# ═══════════════════════════════════════════════════════════════════════════════
out_bio = bio_scores.rename(columns={"predicted_age": "bio_age",
                                      "sex_str": "sex",
                                      "diagnosis_visit": "visit"})[
    ["adrc_id", "visit", "outcome", "diag", "chrono_age",
     "MaxIE_score", "bio_age", "age_gap", "sex"]
]
out_bio.to_csv(os.path.join(RES_ADR, "adrc_bioage_scores.csv"), index=False)
print(f"Saved adrc_bioage_scores.csv  ({len(out_bio)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Save adrc_longitudinal_scores.csv (for brain surface analysis)
# ═══════════════════════════════════════════════════════════════════════════════
out_bio.to_csv(os.path.join(RES_ADR, "adrc_longitudinal_scores.csv"), index=False)
print(f"Saved adrc_longitudinal_scores.csv  ({len(out_bio)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Build lmm_analysis_data.csv
#    Joins AD clock age gap with biomarkers, cognitive scores, and structural MRI
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding LMM analysis data ...")

# AD clock scores only
ad_gap = out_bio[out_bio["outcome"] == "alzheimer_disease"][
    ["adrc_id", "visit", "diag", "sex", "chrono_age", "age_gap"]
].copy()

# Biomarkers
biomarkers = pd.read_csv(os.path.join(DATA_DIR, BIOMARKERS_FILE))
bm_cols = ["adrc_id", "diagnosis_visit", "PTAU217", "PTAU181", "ab_ratio", "GFAP", "NFL"]
biomarkers = biomarkers[[c for c in bm_cols if c in biomarkers.columns]]

# Cognitive scores
cog = pd.read_csv(os.path.join(DATA_DIR, COGNITIVE_FILE))
cog_need = ["adrc_id", "diagnosis_visit",
            "c2_mocatots", "c2_craftvrs", "c2_craftdvr", "c2_animals",
            "c2_digforct", "c2_digbacct", "c2_traila", "c2_trailb", "b4_cdrsum"]
cog = cog[[c for c in cog_need if c in cog.columns]]

# Structural MRI (FreeSurfer volumes)
tau_img = pd.read_csv(os.path.join(DATA_DIR, IMAGING_FILE))
img_id_cols = ["adrc_id", "diagnosis_visit"]

# ICV-normalised volumes
icv_col = "eTIV" if "eTIV" in tau_img.columns else None
hip_l, hip_r   = "Volume_mm3.Left-Hippocampus", "Volume_mm3.Right-Hippocampus"
amyg_l, amyg_r = "Volume_mm3.Left-Amygdala", "Volume_mm3.Right-Amygdala"
vent_ll, vent_rl = "Volume_mm3.Left-Lateral-Ventricle", "Volume_mm3.Right-Lateral-Ventricle"
vent_li, vent_ri = "Volume_mm3.Left-Inf-Lat-Vent", "Volume_mm3.Right-Inf-Lat-Vent"
brain_vol_col = "BrainSegVolNotVent"

lh_thick_cols = [c for c in tau_img.columns
                 if c.startswith("lh_") and c.endswith("_thickness")]

struct = tau_img[img_id_cols].copy()
struct["hippo_vol"]  = (tau_img[hip_l].fillna(0) + tau_img[hip_r].fillna(0)) / 2
struct["amyg_vol"]   = (tau_img[amyg_l].fillna(0) + tau_img[amyg_r].fillna(0)) / 2
struct["vent_vol"]   = (tau_img[vent_ll].fillna(0) + tau_img[vent_rl].fillna(0) +
                        tau_img[vent_li].fillna(0) + tau_img[vent_ri].fillna(0))
struct["brain_vol"]  = tau_img[brain_vol_col] if brain_vol_col in tau_img.columns else np.nan
struct["mean_thick"] = tau_img[lh_thick_cols].mean(axis=1) if lh_thick_cols else np.nan

if icv_col:
    icv = tau_img[img_id_cols + [icv_col]].copy()
    struct = struct.merge(icv, on=img_id_cols, how="left")
    struct["hippo_vol_norm"] = struct["hippo_vol"] / struct["eTIV"].replace(0, np.nan)
    struct["amyg_vol_norm"]  = struct["amyg_vol"]  / struct["eTIV"].replace(0, np.nan)
    struct["vent_vol_norm"]  = struct["vent_vol"]   / struct["eTIV"].replace(0, np.nan)
    struct["brain_vol_norm"] = struct["brain_vol"]  / struct["eTIV"].replace(0, np.nan)
else:
    for col in ["hippo_vol_norm", "amyg_vol_norm", "vent_vol_norm", "brain_vol_norm"]:
        struct[col] = np.nan

struct["wmh_norm"] = np.nan

struct_out = struct[img_id_cols +
                    ["hippo_vol_norm", "amyg_vol_norm", "vent_vol_norm",
                     "wmh_norm", "brain_vol_norm", "mean_thick"]].copy()

# Merge everything
lmm = ad_gap.rename(columns={"visit": "diagnosis_visit"})
for df in [biomarkers, cog, struct_out]:
    df["adrc_id"] = pd.to_numeric(df["adrc_id"], errors="coerce")
    df["diagnosis_visit"] = pd.to_numeric(df["diagnosis_visit"], errors="coerce")
lmm["adrc_id"] = pd.to_numeric(lmm["adrc_id"], errors="coerce")
lmm["diagnosis_visit"] = pd.to_numeric(lmm["diagnosis_visit"], errors="coerce")
lmm = lmm.merge(biomarkers, on=["adrc_id", "diagnosis_visit"], how="left")
lmm = lmm.merge(cog,        on=["adrc_id", "diagnosis_visit"], how="left")
lmm = lmm.merge(struct_out, on=["adrc_id", "diagnosis_visit"], how="left")

print(f"  lmm_analysis_data shape: {lmm.shape}")
print(f"  Subjects: {lmm['adrc_id'].nunique()}")

lmm.to_csv(os.path.join(RES_ADR, "lmm_analysis_data.csv"), index=False)
print(f"Saved lmm_analysis_data.csv  ({len(lmm)} rows)")

print("\nadrc_analysis.py complete.")
