"""
ukb_analysis.py -- Unified analysis pipeline for disease-specific proteomic aging clocks.

Sections:
  1. MaxIE pipeline: cosine test, clock construction, calibration, comparison
  2. Downstream: Cox HR, cumulative incidence, onset timing
  3. Age regression weights: cor(protein, age) for enrichment
  4. Sequential R^2 variance decomposition

Input:  data/   (not provided; see file schema comments below)
Output: results/
"""

import sys, os, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, spearmanr
from numpy.linalg import lstsq
from scipy.optimize import minimize as scipy_min

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))
from optmed import solve_maxie, cosine_test, mediation_stats
from sklearn.metrics import roc_auc_score

DATA    = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ── Data files (not distributed; set your local paths) ───────────────────────
# Plasma proteomics matrix (QC'd, imputed).
#   Columns: IID, Age, Sex, + ~2900 protein columns.
PROTEOMICS_FILE   = "proteomics_matrix.csv"
# Binary case/control labels for 31 diseases.
#   Columns: eid (subject ID), <disease_1> (0/1/NaN), ..., <disease_31>.
CASE_CONTROL_FILE = "case_control_31diseases.csv"
# Train/test split. Columns: IID, split ("train"/"test").
SPLIT_FILE        = "train_test_split.csv"
# Organ-specific aging clock scores (Oh et al. Nature 2023).
#   Columns: IID, <Organ>_pred_age, <Organ>_naive_gap, ...
ORGAN_SCORES_FILE = "organ_clock_scores.csv"
# Time-to-event for 31 diseases.
#   Columns: eid, age_at_baseline, <disease>_event, <disease>_time, <disease>_onset_age.
TTE_FILE          = "time_to_event.csv"
# Metadata columns in the proteomics file (non-protein columns).
META_COLS = ['Barcode', 'IID', 'Visit', 'Sex', 'Age', 'Center', 'APOE4', 'APOE2']
        "and fill in your local data paths. See README for details.")

# ── Disease group mapping ─────────────────────────────────────────────────────
ORGAN_GROUPS = {
    'Brain / Neurological': ['alzheimer_disease', 'all_cause_dementia', 'vascular_dementia',
                              'frontotemporal_dementia', 'amyotrophic_lateral_sclerosis',
                              'parkinson_disease_and_parkinsonism'],
    'Cardiovascular':       ['ischemic_heart_disease', 'hypertensive_disease', 'heart_failure',
                              'atrial_fibrillation_or_flutter', 'cerebrovascular_disease'],
    'Metabolic / Organ':    ['type2_diabetes', 'chronic_kidney_disease', 'chronic_liver_disease'],
    'Respiratory':          ['emphysema_copd'],
    'Cancer':               ['colorectal_cancer', 'lung_cancer', 'breast_cancer', 'prostate_cancer',
                              'leukemia', 'non_hodgkin_lymphoma', 'esophageal_cancer',
                              'liver_cancer', 'pancreatic_cancer', 'brain_cancer', 'ovarian_cancer'],
    'Musculoskeletal':      ['rheumatoid_arthritis', 'osteoporosis', 'osteoarthritis'],
    'Ophthalmic':           ['macular_degeneration'],
    'Psychiatric':          ['depression'],
}
LABEL_TO_GROUP = {lab: grp for grp, labs in ORGAN_GROUPS.items() for lab in labs}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_fstar(M, A, Y):
    res = mediation_stats(M, A, Y)
    return float(res.loc['mediation_index', 'Value'])

def compute_gap(M, A):
    A_c = A - A.mean(); M_c = M - M.mean()
    alpha = float(M_c @ A_c) / float(A_c @ A_c + 1e-15)
    return M_c - alpha * A_c

def safe_auc(Y, S):
    try: return round(roc_auc_score(Y, S), 4)
    except: return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MaxIE pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: MaxIE pipeline")
print("=" * 70)

print("Loading protein matrix...")
prot_raw = pd.read_csv(DATA / PROTEOMICS_FILE)
PROTEIN_COLS = [c for c in prot_raw.columns if c not in META_COLS]
p = len(PROTEIN_COLS)
print(f"  {len(prot_raw)} samples, {p} proteins")

cc = pd.read_csv(DATA / CASE_CONTROL_FILE).rename(columns={'eid': 'IID'})
disease_cols = [c for c in cc.columns if c != 'IID']
print(f"  {len(cc)} participants, {len(disease_cols)} diseases")

merged_all = prot_raw.merge(cc, on='IID', how='inner').dropna(subset=['Age'] + PROTEIN_COLS)
print(f"  Merged: {len(merged_all)} participants")

# Train/test split
split_df = pd.read_csv(DATA / SPLIT_FILE)
iid_split = split_df.drop_duplicates('IID')[['IID', 'split']]
merged_all = merged_all.merge(iid_split, on='IID', how='inner')
tr = merged_all[merged_all['split'] == 'train']
te = merged_all[merged_all['split'] == 'test']
print(f"  Train: {len(tr)}, Test: {len(te)}")

# Z-score using train statistics (frozen -- save for application to new individuals)
X_tr_all = tr[PROTEIN_COLS].values.astype(float)
mu_tr = X_tr_all.mean(axis=0)
sd_tr = X_tr_all.std(axis=0); sd_tr[sd_tr < 1e-10] = 1.0
pd.DataFrame({'protein': PROTEIN_COLS, 'mu': mu_tr, 'sd': sd_tr}).to_csv(
    RESULTS / 'normalization_params.csv', index=False)
print(f"  Saved normalization_params.csv ({len(PROTEIN_COLS)} proteins)")
X_all_z = (merged_all[PROTEIN_COLS].values.astype(float) - mu_tr) / sd_tr
for i, col in enumerate(PROTEIN_COLS):
    merged_all[col] = X_all_z[:, i]
tr = merged_all[merged_all['split'] == 'train']
te = merged_all[merged_all['split'] == 'test']

# Organ clock comparison (exclude Conventional, Organismal, sex-specific)
organ_scores = pd.read_csv(DATA / ORGAN_SCORES_FILE)
ORGAN_COLS_PRED = [c for c in organ_scores.columns if c.endswith('_pred_age')
                   and not c.startswith('Female') and not c.startswith('Male')
                   and not c.startswith('Conventional') and not c.startswith('Organismal')]
organ_names = [c.replace('_pred_age', '') for c in ORGAN_COLS_PRED]

# Main loop
summary_rows, all_bioage, all_sexgaps, all_or_rows, comp_rows = [], [], [], [], []
all_weights = {}
calib_rows = []

print()
for disease in disease_cols:
    Y_tr_raw = tr[disease].values.astype(float)
    Y_te_raw = te[disease].values.astype(float)
    tr_valid = ~np.isnan(Y_tr_raw); te_valid = ~np.isnan(Y_te_raw)
    A_tr = tr['Age'].values[tr_valid]; A_te = te['Age'].values[te_valid]
    X_tr = tr[PROTEIN_COLS].values[tr_valid]; X_te = te[PROTEIN_COLS].values[te_valid]
    Y_tr = Y_tr_raw[tr_valid]; Y_te = Y_te_raw[te_valid]
    Sex_tr = tr['Sex'].values[tr_valid]; Sex_te = te['Sex'].values[te_valid]
    IID_te = te['IID'].values[te_valid]
    n_cases_te = int(Y_te.sum())
    if int(Y_tr.sum()) < 10 or n_cases_te < 10:
        continue

    # Cosine test (all samples)
    Y_all = merged_all[disease].values.astype(float); valid_all = ~np.isnan(Y_all)
    ct = cosine_test(merged_all[PROTEIN_COLS].values[valid_all],
                     merged_all['Age'].values[valid_all], Y_all[valid_all])
    cos_phi, pval = ct['cos_phi'], ct['p_value']

    # MaxIE
    try: w = solve_maxie(X_tr, A_tr, Y_tr)[0]
    except: continue

    M_te = X_te @ w
    f_m = get_fstar(M_te, A_te, Y_te)
    r_MA = float(np.corrcoef(M_te, A_te)[0, 1])
    auc_m = safe_auc(Y_te, M_te)

    # Calibration: delta = (M - mu0 - alpha*A - delta_sex*S) / alpha
    M_tr = X_tr @ w

    design_m = np.column_stack([np.ones(len(A_tr)), A_tr, Sex_tr])
    coefs_m = lstsq(design_m, M_tr, rcond=None)[0]
    b0, b1, b2 = float(coefs_m[0]), float(coefs_m[1]), float(coefs_m[2])

    delta_cal = (M_te - b0 - A_te * b1 - Sex_te * b2) / (b1 + 1e-15)
    pred_age  = A_te + delta_cal
    sex_gap   = delta_cal  # sex centred at zero by construction

    calib_rows.append({'outcome': disease, 'b0': b0, 'b1': b1, 'b2': b2})

    all_bioage.append(pd.DataFrame({
        'IID': IID_te, 'outcome': disease, 'chrono_age': A_te,
        'MaxIE_score': M_te, 'predicted_age': pred_age, 'age_gap': delta_cal}))
    all_sexgaps.append(pd.DataFrame({
        'IID': IID_te, 'outcome': disease, 'chrono_age': A_te,
        'sex': Sex_te, 'age_gap_naive': delta_cal, 'age_gap_sex_adj': sex_gap}))
    all_weights[disease] = w

    # Best organ clock
    te_iid_df = pd.DataFrame({'IID': IID_te, '_idx': range(len(IID_te))})
    org_merged = te_iid_df.merge(organ_scores, on='IID', how='inner').sort_values('_idx')
    organ_fstars = {}
    if len(org_merged) > 100:
        idx_org = org_merged['_idx'].values
        A_org = A_te[idx_org]
        Y_org = Y_te[idx_org]
        for org_name, org_col in zip(organ_names, ORGAN_COLS_PRED):
            if org_col in org_merged.columns:
                try: organ_fstars[org_name] = get_fstar(org_merged[org_col].values, A_org, Y_org)
                except: pass
    best_org = max(organ_fstars, key=organ_fstars.get) if organ_fstars else 'None'
    best_org_fstar = organ_fstars.get(best_org, 0.0)

    # Fair comparison
    best_org_col = f'{best_org}_pred_age'
    if len(org_merged) > 100 and best_org != 'None' and best_org_col in org_merged.columns:
        idx_v = org_merged['_idx'].values
        A_fc, Y_fc, M_maxie = A_te[idx_v], Y_te[idx_v], M_te[idx_v]
        M_organ = org_merged[best_org_col].values
        if Y_fc.sum() >= 20 and (1 - Y_fc).sum() >= 20:
            comp_rows.append({
                'outcome': disease, 'group': LABEL_TO_GROUP.get(disease, 'Other'),
                'best_organ': best_org,
                'r_age_maxie': round(float(np.corrcoef(M_maxie, A_fc)[0,1]), 4),
                'r_age_organ': round(float(np.corrcoef(M_organ, A_fc)[0,1]), 4),
                'auc_age': safe_auc(Y_fc, A_fc),
                'auc_full_maxie': safe_auc(Y_fc, M_maxie),
                'auc_full_organ': safe_auc(Y_fc, M_organ),
            })

    # OR per year
    bio_age_te = A_te + sex_gap
    Xlr = np.column_stack([np.ones(len(Y_te)), bio_age_te, A_te])
    def neg_ll(b):
        eta = np.clip(Xlr @ b, -30, 30); pp = 1/(1+np.exp(-eta))
        pp = np.clip(pp, 1e-15, 1-1e-15)
        return -np.sum(Y_te*np.log(pp) + (1-Y_te)*np.log(1-pp))
    try:
        res_lr = scipy_min(neg_ll, np.zeros(3), method='Newton-CG',
                          jac=lambda b: Xlr.T @ (1/(1+np.exp(-Xlr@b)) - Y_te), options={'maxiter': 200})
        g1 = res_lr.x[1]; eta = Xlr @ res_lr.x; pp = 1/(1+np.exp(-eta))
        se1 = np.sqrt(np.diag(np.linalg.inv(Xlr.T @ (Xlr * (pp*(1-pp))[:,None]))))[1]
        OR_yr = np.exp(g1); OR_lo = np.exp(g1-1.96*se1); OR_hi = np.exp(g1+1.96*se1)
        p_or = 2*norm.sf(abs(g1/se1))
    except:
        OR_yr = OR_lo = OR_hi = p_or = np.nan

    all_or_rows.append({'outcome': disease, 'n_cases': n_cases_te,
                        'OR_per_yr': round(OR_yr,4) if not np.isnan(OR_yr) else np.nan,
                        'OR_lo': round(OR_lo,4) if not np.isnan(OR_lo) else np.nan,
                        'OR_hi': round(OR_hi,4) if not np.isnan(OR_hi) else np.nan, 'p_M': p_or})

    summary_rows.append({
        'outcome': disease, 'group': LABEL_TO_GROUP.get(disease, 'Other'),
        'n_test': int(te_valid.sum()), 'n_cases_test': n_cases_te,
        'cos_phi': round(cos_phi, 4), 'p_value': pval,
        'MaxIE_f_star': round(f_m, 4), 'MaxIE_r_M_Age': round(r_MA, 4),
        'MaxIE_AUC': round(auc_m, 4) if not np.isnan(auc_m) else np.nan,
        'best_organ': best_org, 'Organ_f_star': round(best_org_fstar, 4),
    })

    sig = '*' if pval < 0.05/len(disease_cols) else ''
    print(f"  {disease:45s} cases={n_cases_te:5d}  p={pval:.1e}{sig:2s}  f*={f_m:.4f}  AUC={auc_m:.3f}")

# Save Section 1 outputs
print("\nSaving Section 1...")
pd.DataFrame(summary_rows).sort_values('p_value').to_csv(RESULTS / 'disease_summary.csv', index=False)
pd.DataFrame(comp_rows).to_csv(RESULTS / 'fair_comparison.csv', index=False)
pd.DataFrame(calib_rows).to_csv(RESULTS / 'calibration_params.csv', index=False)
print(f"  Saved calibration_params.csv ({len(calib_rows)} diseases)")
pd.concat(all_bioage, ignore_index=True).to_csv(RESULTS / 'bioage_scores.csv', index=False)
pd.concat(all_sexgaps, ignore_index=True).to_csv(RESULTS / 'sex_adjusted_gaps.csv', index=False)
pd.DataFrame(all_or_rows).to_csv(RESULTS / 'logistic_or.csv', index=False)

# Weight matrix
w_df = pd.DataFrame({'protein': PROTEIN_COLS})
for dis, w in all_weights.items():
    w_df[dis] = w
w_df.to_csv(RESULTS / 'all_weights.csv', index=False)
print(f"  Saved {len(all_weights)} disease weights")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Downstream -- Cox HR, KM, onset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Downstream analyses")
print("=" * 70)

from lifelines import CoxPHFitter, KaplanMeierFitter

sex_gaps = pd.read_csv(RESULTS / 'sex_adjusted_gaps.csv')
bioage = pd.read_csv(RESULTS / 'bioage_scores.csv')
tte = pd.read_csv(DATA / TTE_FILE)

scores = sex_gaps[['IID','outcome','chrono_age','age_gap_sex_adj']].copy()
scores['bio_age'] = scores['chrono_age'] + scores['age_gap_sex_adj']
scores = scores.merge(bioage[['IID','outcome','MaxIE_score']], on=['IID','outcome'], how='left')

tte_event_cols = [c.replace('_event','') for c in tte.columns if c.endswith('_event')]
cox_rows, ci_rows, onset_rows = [], [], []

for tte_prefix in tte_event_cols:
    event_col, time_col, onset_col = f'{tte_prefix}_event', f'{tte_prefix}_time', f'{tte_prefix}_onset_age'
    our_name = tte_prefix
    if our_name not in scores['outcome'].unique(): continue
    sc = scores[scores['outcome'] == our_name]
    if len(sc) == 0: continue

    tte_sub = tte[['eid','age_at_baseline',event_col,time_col,onset_col]].rename(columns={'eid':'IID'})
    merged = sc.merge(tte_sub, on='IID').dropna(subset=[event_col,time_col])
    prev = merged[onset_col].notna() & (merged[onset_col] <= merged['age_at_baseline'])
    merged = merged[~prev].copy()
    merged['duration'] = merged[time_col] - merged['age_at_baseline']
    merged = merged[merged['duration'] > 0].copy()
    merged['event'] = merged[event_col].astype(int)
    n_events = merged['event'].sum()
    if n_events < 20: continue

    # Cox HR
    try:
        cph = CoxPHFitter()
        cph.fit(merged[['duration','event','bio_age','chrono_age']], 'duration', 'event')
        hr = np.exp(cph.params_['bio_age'])
        hr_lo = np.exp(cph.confidence_intervals_.loc['bio_age','95% lower-bound'])
        hr_hi = np.exp(cph.confidence_intervals_.loc['bio_age','95% upper-bound'])
        cox_rows.append({'outcome': our_name, 'n_incident': int(n_events),
                         'HR_per_yr': round(hr,4), 'HR_lo': round(hr_lo,4),
                         'HR_hi': round(hr_hi,4), 'p_M': cph.summary.loc['bio_age','p']})
        print(f"  {our_name:45s} events={n_events:5d}  HR={hr:.3f}")
    except Exception as e:
        print(f"  {our_name:45s} Cox failed: {e}")

    # KM cumulative incidence
    for strat_name, strat_col in [('age_gap','age_gap_sex_adj'), ('raw_M','MaxIE_score')]:
        if strat_col not in merged.columns: continue
        lo = np.percentile(merged[strat_col].dropna(), 10)
        hi = np.percentile(merged[strat_col].dropna(), 90)
        merged['bio_group'] = 'Normal'
        merged.loc[merged[strat_col] < lo, 'bio_group'] = 'Young'
        merged.loc[merged[strat_col] > hi, 'bio_group'] = 'Old'
        for grp in ['Young','Normal','Old']:
            sub = merged[merged['bio_group']==grp]
            if len(sub) < 20: continue
            kmf = KaplanMeierFitter(); kmf.fit(sub['duration'], sub['event'])
            ci_func = kmf.confidence_interval_cumulative_density_
            for t in np.arange(0, min(sub['duration'].max(), 16), 0.25):
                surv = kmf.predict(t)
                try:
                    idx = max(0, min(ci_func.index.searchsorted(t, side='right')-1, len(ci_func)-1))
                    ci_lo_v, ci_hi_v = ci_func.iloc[idx,0]*100, ci_func.iloc[idx,1]*100
                except: ci_lo_v = ci_hi_v = np.nan
                ci_rows.append({'outcome': our_name, 'stratification': strat_name,
                                'bio_group': grp, 'time': round(t,2),
                                'cumulative_incidence': round((1-surv)*100,4),
                                'ci_lo': round(ci_lo_v,4), 'ci_hi': round(ci_hi_v,4)})

    # Onset timing
    cases = merged[merged['event']==1].copy()
    cases['time_to_onset'] = cases[onset_col] - cases['age_at_baseline']
    cases = cases[cases['time_to_onset'] > 0]
    for _, row in cases.iterrows():
        onset_rows.append({'outcome': our_name, 'age_gap': row['age_gap_sex_adj'],
                           'time_to_onset': row['time_to_onset']})

print("\nSaving Section 2...")
pd.DataFrame(cox_rows).to_csv(RESULTS / 'cox_hr.csv', index=False)
pd.DataFrame(ci_rows).to_csv(RESULTS / 'cumulative_incidence.csv', index=False)
pd.DataFrame(onset_rows).to_csv(RESULTS / 'agegap_vs_onset.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Age regression weights
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Age regression weights")
print("=" * 70)

print("  Using train-split z-scored proteins...")
X_z = tr[PROTEIN_COLS].values.astype(float)
age = tr['Age'].values
age_z = (age - np.mean(age)) / np.std(age)
r_age = X_z.T @ age_z / len(age_z)

pd.DataFrame({'protein': PROTEIN_COLS, 'weight': r_age}).to_csv(
    RESULTS / 'age_regression_weights.csv', index=False)
print(f"  Saved age_regression_weights.csv (top: {[PROTEIN_COLS[i] for i in np.argsort(r_age)[-3:][::-1]]})")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Sequential R^2 variance decomposition
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Sequential R^2 variance decomposition")
print("=" * 70)

organ_df = pd.read_csv(DATA / ORGAN_SCORES_FILE)

ALL_GAP_COLS   = [c for c in organ_df.columns if c.endswith('_naive_gap')
                  and 'Female' not in c and 'Male' not in c]
CONV_GAP_COLS  = ['Conventional_naive_gap', 'Organismal_naive_gap']
ORG_GAP_COLS   = [c for c in ALL_GAP_COLS if c not in CONV_GAP_COLS]
print(f"  Conventional clocks: {CONV_GAP_COLS}")
print(f"  Organ-specific clocks ({len(ORG_GAP_COLS)}): {ORG_GAP_COLS[:3]} ...")

bioage = pd.read_csv(RESULTS / 'bioage_scores.csv')
merged_all = bioage.merge(
    organ_df[['IID'] + ALL_GAP_COLS], on='IID', how='inner')

def seq_r2_4(y, age, conv_gaps, org_gaps):
    """Sequential R^2 for age -> age+conv -> age+all -> residual."""
    def r2(X_cols):
        X = np.column_stack([np.ones(len(y))] + X_cols)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 50:
            return np.nan
        Xm, ym = X[mask], y[mask]
        coef = np.linalg.lstsq(Xm, ym, rcond=None)[0]
        ss_res = np.sum((ym - Xm @ coef)**2)
        ss_tot = np.sum((ym - ym.mean())**2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    r1 = r2([age])
    r2v = r2([age] + [conv_gaps[:, i] for i in range(conv_gaps.shape[1])])
    r3 = r2([age] + [conv_gaps[:, i] for i in range(conv_gaps.shape[1])]
               + [org_gaps[:, i] for i in range(org_gaps.shape[1])])
    return r1, max(0.0, r2v - r1), max(0.0, r3 - r2v), max(0.0, 1.0 - r3)

hier_rows = []
for disease in sorted(merged_all['outcome'].unique()):
    dm = merged_all[merged_all['outcome'] == disease].copy()
    y         = dm['predicted_age'].values.astype(float)
    age_v     = dm['chrono_age'].values.astype(float)
    conv_mat  = dm[CONV_GAP_COLS].values.astype(float)
    org_mat   = dm[ORG_GAP_COLS].values.astype(float)
    r1, rc, ro, rn = seq_r2_4(y, age_v, conv_mat, org_mat)
    hier_rows.append({'outcome': disease,
                      'pct_age':          round(r1*100, 2),
                      'pct_conventional': round(rc*100, 2),
                      'pct_organ':        round(ro*100, 2),
                      'pct_new':          round(rn*100, 2)})
    print(f"  {disease:40s}  age={r1*100:.1f}  conv={rc*100:.1f}  "
          f"organ={ro*100:.1f}  new={rn*100:.1f}")

pd.DataFrame(hier_rows).to_csv(RESULTS / 'paper_variance_hierarchy_4level.csv', index=False)
print(f"  Saved paper_variance_hierarchy_4level.csv ({len(hier_rows)} diseases)")

hier3 = pd.DataFrame(hier_rows).copy()
hier3['pct_organ'] = hier3['pct_conventional'] + hier3['pct_organ']
hier3 = hier3.drop(columns='pct_conventional')
hier3.to_csv(RESULTS / 'paper_variance_hierarchy_clean.csv', index=False)
print(f"  Saved paper_variance_hierarchy_clean.csv")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
