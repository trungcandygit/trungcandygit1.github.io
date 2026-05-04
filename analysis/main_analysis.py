"""
Dynamic Factor Analysis on VN30/HOSE Stock Returns
Reproduces Figures 1-10 and Tables 1-7 from paper 2510.15938v1.pdf
Vietnam HOSE replaces Philippine PSE.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL IDENTITY
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'F1': '#1B4F8A', 'F2': '#E67E22', 'PC1': '#27AE60', 'PC2': '#8E44AD',
    'INDEX': '#2C3E50', 'GDP': '#1B4F8A', 'AR1': '#7F8C8D', 'AR1_F': '#27AE60',
    'SCATTER1': '#1B4F8A', 'SCATTER2': '#E67E22', 'VLINE': '#C0392B',
    'HEATMAP': 'RdYlBu_r', 'BAR_POS': '#1B4F8A', 'BAR_NEG': '#C0392B',
}
mpl.rcParams.update({
    'figure.dpi': 150, 'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.4, 'grid.linestyle': '--', 'grid.linewidth': 0.5,
    'lines.linewidth': 1.5, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.framealpha': 0.8,
    'legend.edgecolor': '#CCCCCC',
})

OUTFIG = 'output/figures'
OUTTAB = 'output/tables'
os.makedirs(OUTFIG, exist_ok=True)
os.makedirs(OUTTAB, exist_ok=True)

def scale_series(s):
    arr = np.asarray(s, dtype=float)
    return (arr - np.nanmean(arr)) / np.nanstd(arr)

def add_sig_stars(pval):
    if pval < 0.01:  return '***'
    if pval < 0.05:  return '**'
    if pval < 0.10:  return '*'
    return 'ns'

def fmt_date_axis(ax, dates):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
returns_raw = pd.read_csv('data/returns.csv', index_col='date', parse_dates=True)
index_df    = pd.read_csv('data/index.csv',   index_col='date', parse_dates=True)
macro_df    = pd.read_csv('data/macro_wb.csv',index_col='date', parse_dates=True)

# Clean returns: keep stocks with ≥40% coverage (to include stocks listed post-2015)
thresh = int(0.40 * len(returns_raw))
returns = returns_raw.dropna(axis=1, thresh=thresh).copy()

# Forward-fill THEN backward-fill to cover pre-listing NaN with first available return
# Then fill any remaining NaN with 0 (neutral return)
returns = returns.ffill().bfill().fillna(0)
# Drop rows where ALL stocks are NaN (only the very first row from pct_change)
returns = returns.dropna(how='all')

tickers = returns.columns.tolist()
dates   = returns.index
T, S    = returns.shape
print(f"  Returns: {T} months × {S} stocks  ({dates[0].date()} – {dates[-1].date()})")

# VN-Index return (align to returns index)
vnindex_ret  = index_df['VNINDEX_return'].reindex(dates).fillna(0)
vnindex_close= index_df['VNINDEX_close'].reindex(dates).ffill()

# Risk-free rate (monthly, %)  rf_m = annual_yield / 12 / 100
rf_monthly = (macro_df['risk_free_rate'].reindex(dates).ffill() / 12 / 100).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INFORMATION CRITERIA — Bai & Ng (2002)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing IC...")
N, Tt = S, T
ic1_vals, ic2_vals, ic3_vals = [], [], []

for r in range(6):
    if r == 0:
        V = returns.values.var(axis=0).mean()
    else:
        pca_tmp = PCA(n_components=r).fit(returns.values)
        resid   = returns.values - pca_tmp.inverse_transform(pca_tmp.transform(returns.values))
        V       = np.mean(resid ** 2)
    log_V = np.log(max(V, 1e-12))
    g1 = r * (N + Tt) / (N * Tt) * np.log((N * Tt) / (N + Tt))
    g2 = r * (N + Tt) / (N * Tt) * np.log(min(N, Tt))
    g3 = r * np.log(min(N, Tt)) / min(N, Tt)
    ic1_vals.append(log_V + g1)
    ic2_vals.append(log_V + g2)
    ic3_vals.append(log_V + g3)

print(f"  IC minimised at n = {np.argmin(ic1_vals)} (IC1), "
      f"{np.argmin(ic2_vals)} (IC2), {np.argmin(ic3_vals)} (IC3)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA (n=1, n=2) for comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Fitting PCA...")
pca2  = PCA(n_components=2).fit(returns.values)
pc_all = pca2.transform(returns.values)
PC1   = pd.Series(pc_all[:, 0], index=dates)
PC2   = pd.Series(pc_all[:, 1], index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT DFM n=1  (k_factors=1, factor_lag=0, factor_order=3, error_order=0)
# ─────────────────────────────────────────────────────────────────────────────
from dynamicfactoranalysis import DynamicFactorModel

print("Fitting DFM n=1 (this may take 1-2 minutes)...")
dfm1 = DynamicFactorModel(
    returns,           # pass DataFrame so endog_names = tickers
    k_factors=1,
    factor_lag=0,
    factor_order=3,
    error_order=0,
)
res1 = dfm1.fit(method='powell', disp=False)
print("  DFM n=1 fitted.")

# Extract factor (smoothed)
F1t = pd.Series(res1.factors.smoothed[0], index=dates)

# Flip sign so Factor 1 correlates positively with VN-Index
if np.corrcoef(F1t.values, vnindex_ret.values)[0, 1] < 0:
    F1t = -F1t

# Extract loadings β_i (one per stock, factor_lag=0 → single value per stock)
load_params1 = res1.params[dfm1._params_loadings]   # shape (S,)
# param order: for each stock i, then each lag j, then each factor k
# With factor_lag=0, k_factors=1: one value per stock
beta1 = load_params1.values if hasattr(load_params1, 'values') else np.array(load_params1)
# Flip loadings sign consistently with factor
if np.corrcoef(F1t.values, vnindex_ret.values)[0, 1] < 0:
    beta1 = -beta1

# Extract σ_i = sqrt(error variance)
sigma1 = np.sqrt(np.abs(res1.params[dfm1._params_error_cov].values
                         if hasattr(res1.params[dfm1._params_error_cov], 'values')
                         else np.array(res1.params[dfm1._params_error_cov])))

# AR(3) transition coefficients
ar1_params = (res1.params[dfm1._params_factor_transition].values
              if hasattr(res1.params[dfm1._params_factor_transition], 'values')
              else np.array(res1.params[dfm1._params_factor_transition]))

print(f"  F1t corr w/ VNIndex: {np.corrcoef(F1t.values, vnindex_ret.values)[0,1]:.4f}")
print(f"  F1t corr w/ PC1:     {np.corrcoef(F1t.values, PC1.values)[0,1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT DFM n=2  (k_factors=2, factor_lag=0, factor_order=2, error_order=0)
# ─────────────────────────────────────────────────────────────────────────────
print("Fitting DFM n=2 (this may take 2-3 minutes)...")
dfm2 = DynamicFactorModel(
    returns,
    k_factors=2,
    factor_lag=0,
    factor_order=2,
    error_order=0,
)
res2 = dfm2.fit(method='powell', disp=False)
print("  DFM n=2 fitted.")

# Extract factors (smoothed)
F1t_2 = pd.Series(res2.factors.smoothed[0], index=dates)
F2t_2 = pd.Series(res2.factors.smoothed[1], index=dates)

# Identify: F1 ~ market (positive correlation with VNIndex)
if np.corrcoef(F1t_2.values, vnindex_ret.values)[0, 1] < 0:
    F1t_2 = -F1t_2
if np.corrcoef(F2t_2.values, vnindex_ret.values)[0, 1] < 0:
    F2t_2 = -F2t_2

# Extract loadings: params order for n=2, factor_lag=0:
# stock0_f1, stock0_f2, stock1_f1, stock1_f2, ...
load_params2_raw = (res2.params[dfm2._params_loadings].values
                    if hasattr(res2.params[dfm2._params_loadings], 'values')
                    else np.array(res2.params[dfm2._params_loadings]))
b1_arr = load_params2_raw[0::2]   # β_1i for each stock
b2_arr = load_params2_raw[1::2]   # β_2i for each stock
load2  = np.column_stack([b1_arr, b2_arr])   # shape (S, 2)

# Flip consistently with factor direction
if np.corrcoef(F1t_2.values, vnindex_ret.values)[0, 1] < 0:
    load2[:, 0] = -load2[:, 0]
if np.corrcoef(F2t_2.values, vnindex_ret.values)[0, 1] < 0:
    load2[:, 1] = -load2[:, 1]

sigma2 = np.sqrt(np.abs(
    res2.params[dfm2._params_error_cov].values
    if hasattr(res2.params[dfm2._params_error_cov], 'values')
    else np.array(res2.params[dfm2._params_error_cov])
))

var2_params = (res2.coefficient_matrices_var  # shape (factor_order, k_factors, k_factors)
               if res2.coefficient_matrices_var is not None else np.zeros((2, 2, 2)))

print(f"  F1t_2 corr w/ VNIndex: {np.corrcoef(F1t_2.values, vnindex_ret.values)[0,1]:.4f}")
print(f"  F2t_2 corr w/ VNIndex: {np.corrcoef(F2t_2.values, vnindex_ret.values)[0,1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CAPM BETAS
# ─────────────────────────────────────────────────────────────────────────────
print("Computing CAPM betas...")
excess_mkt = vnindex_ret - rf_monthly * 100   # convert rf to % for returns in %
capm_beta, capm_pval, capm_alpha = {}, {}, {}
for col in tickers:
    y = returns[col] - rf_monthly * 100
    idx_valid = (~y.isna()) & (~excess_mkt.isna())
    sl, ic, r, p, se = stats.linregress(excess_mkt[idx_valid], y[idx_valid])
    capm_beta[col]  = sl
    capm_pval[col]  = p
    capm_alpha[col] = ic

capm_vals = np.array([capm_beta[t] for t in tickers])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Information Criteria
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 1...")
fig, ax = plt.subplots(figsize=(5.5, 4))
ns = list(range(6))
ax.plot(ns, ic1_vals, color=COLORS['F1'],  label='IC$_1$', marker='o', ms=5)
ax.plot(ns, ic2_vals, color=COLORS['F2'],  label='IC$_2$', marker='s', ms=5)
ax.plot(ns, ic3_vals, color=COLORS['PC1'], label='IC$_3$', marker='^', ms=5)
ax.axvline(np.argmin(ic2_vals), color=COLORS['VLINE'], lw=0.8,
           linestyle=':', alpha=0.7, label=f'min IC$_2$ at n={np.argmin(ic2_vals)}')
ax.set_xlabel('Number of Common Factors $n$')
ax.set_ylabel('Information Criterion')
ax.set_title('Bai-Ng (2002) Information Criteria — VN30/HOSE')
ax.legend(loc='upper left')
ax.set_xticks(ns)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.1 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — n=1: Smoothed F1t | PC1 | VN-Index (3 subplots)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 2...")
corr_ft_idx = np.corrcoef(scale_series(F1t), vnindex_ret.fillna(0))[0, 1]
corr_ft_pc1 = np.corrcoef(scale_series(F1t), scale_series(PC1))[0, 1]
corr_pc1_idx= np.corrcoef(scale_series(PC1), vnindex_ret.fillna(0))[0, 1]

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ylim = (-4.5, 4.5)

ax = axes[0]
ax.plot(dates, scale_series(F1t), color=COLORS['F1'], lw=1.2)
ax.set_title(f'Scaled Smoothed $F_t$ (DFM $n=1$, $p=3$)')
ax.set_ylim(ylim); ax.axhline(0, color='gray', lw=0.5)
ax.text(0.01, 0.05, f'Corr(Ft, VNI)={corr_ft_idx:.3f}  Corr(Ft, PC1)={corr_ft_pc1:.3f}',
        transform=ax.transAxes, fontsize=8, color='dimgray')

ax = axes[1]
ax.plot(dates, scale_series(PC1), color=COLORS['PC1'], lw=1.2)
ax.set_title('Scaled Principal Component 1')
ax.set_ylim(ylim); ax.axhline(0, color='gray', lw=0.5)
ax.text(0.01, 0.05, f'Corr(PC1, VNI)={corr_pc1_idx:.3f}',
        transform=ax.transAxes, fontsize=8, color='dimgray')

ax = axes[2]
ax.plot(dates, vnindex_ret * 100, color=COLORS['INDEX'], lw=0.8)
ax.set_title('VN-Index Monthly Returns (%)')
ax.set_ylim(-20, 20); ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Year')

for ax in axes:
    fmt_date_axis(ax, dates)

plt.suptitle('DFM n=1 — VN30/HOSE (2015–2023)', y=1.01, fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.2 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Scatterplot β_i vs CAPM beta (n=1)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 3...")
corr13 = np.corrcoef(beta1, capm_vals)[0, 1]
m3, b3, r3, p3, _ = stats.linregress(beta1, capm_vals)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(beta1, capm_vals, color=COLORS['SCATTER1'], s=35, alpha=0.85, edgecolors='none')
for i, t in enumerate(tickers):
    ax.annotate(t, (beta1[i], capm_vals[i]), fontsize=5.5,
                xytext=(2, 2), textcoords='offset points', color='#555555')
xline = np.linspace(beta1.min(), beta1.max(), 100)
ax.plot(xline, m3 * xline + b3, color=COLORS['VLINE'], lw=1, linestyle='--', alpha=0.7)
ax.set_xlabel(r'DFM Factor Loading $\beta_i$')
ax.set_ylabel(r'CAPM $\hat{\beta}_i$')
ax.set_title(r'DFM $\beta_i$ vs CAPM $\hat{\beta}_i$ — $n=1$')
ax.text(0.05, 0.92, f'Corr = {corr13:.4f}{add_sig_stars(p3)}',
        transform=ax.transAxes, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure3.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.3 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — n=2: F1t | PC1 | F2t | PC2 | VN-Index (5 subplots)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 4...")
corr_f1_idx2 = np.corrcoef(scale_series(F1t_2), vnindex_ret.fillna(0))[0, 1]
corr_f2_idx2 = np.corrcoef(scale_series(F2t_2), vnindex_ret.fillna(0))[0, 1]
corr_f1_pc1  = np.corrcoef(scale_series(F1t_2), scale_series(PC1))[0, 1]
corr_f2_pc2  = np.corrcoef(scale_series(F2t_2), scale_series(PC2))[0, 1]

series_list = [
    (scale_series(F1t_2), COLORS['F1'],    f'Scaled $F_{{1t}}$ (DFM $n=2$)  [Corr VNI={corr_f1_idx2:.3f}]'),
    (scale_series(PC1),   COLORS['PC1'],   f'Scaled PC1  [Corr VNI={np.corrcoef(scale_series(PC1), vnindex_ret.fillna(0))[0,1]:.3f}]'),
    (scale_series(F2t_2), COLORS['F2'],    f'Scaled $F_{{2t}}$ (DFM $n=2$)  [Corr VNI={corr_f2_idx2:.3f}]'),
    (scale_series(PC2),   COLORS['PC2'],   f'Scaled PC2'),
    (vnindex_ret * 100,   COLORS['INDEX'], 'VN-Index Returns (%)'),
]
fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
for ax, (ser, col, title) in zip(axes, series_list):
    vals = ser.values if hasattr(ser, 'values') else np.array(ser)
    ax.plot(dates, vals, color=col, lw=0.9)
    ax.set_title(title)
    ax.set_ylim(-20, 20); ax.axhline(0, color='gray', lw=0.5)
    fmt_date_axis(ax, dates)
axes[-1].set_xlabel('Year')
plt.suptitle('DFM n=2 — VN30/HOSE (2015–2023)', y=1.01, fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure4.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.4 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — β1i, β2i vs CAPM (n=2)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 5...")
b1 = load2[:, 0]; b2 = load2[:, 1]
c1 = np.corrcoef(b1, capm_vals)[0, 1]
c2 = np.corrcoef(b2, capm_vals)[0, 1]
m5a, b5a, _, p5a, _ = stats.linregress(b1, capm_vals)
m5b, b5b, _, p5b, _ = stats.linregress(b2, capm_vals)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(b1, capm_vals, color=COLORS['SCATTER1'], s=30, alpha=0.8,
           edgecolors='none', label=rf'$\beta_{{1i}}$ (Corr={c1:.3f}{add_sig_stars(p5a)})')
ax.scatter(b2, capm_vals, color=COLORS['SCATTER2'], s=30, alpha=0.8,
           edgecolors='none', label=rf'$\beta_{{2i}}$ (Corr={c2:.3f}{add_sig_stars(p5b)})')
for bvals, col, m, bv in [(b1, COLORS['SCATTER1'], m5a, b5a), (b2, COLORS['SCATTER2'], m5b, b5b)]:
    xl = np.linspace(bvals.min(), bvals.max(), 100)
    ax.plot(xl, m * xl + bv, color=col, lw=1, linestyle='--', alpha=0.6)
ax.set_xlabel(r'$\beta_i$')
ax.set_ylabel(r'CAPM $\hat{\beta}_i$')
ax.set_title(r'DFM $\beta_{1i}$, $\beta_{2i}$ vs CAPM $\hat{\beta}_i$ — $n=2$')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure5.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.5 done")


# ─────────────────────────────────────────────────────────────────────────────
# GDP NOWCASTING — preparation
# ─────────────────────────────────────────────────────────────────────────────
print("GDP nowcasting setup...")
gdp_raw = pd.read_csv('../data/csv/macro_gdp_quarterly.csv')
gdp_raw['date'] = pd.to_datetime(gdp_raw['date'])
gdp_q = gdp_raw.set_index('date')['GDP_growth_pct_yoy']
gdp_q.index = gdp_q.index.to_period('Q').to_timestamp('Q')

# Build quarterly factor aggregates (mean within each quarter)
fac_df = pd.DataFrame({'F1': F1t_2.values, 'F2': F2t_2.values}, index=dates)
fac_q  = fac_df.resample('QE').mean()
fac_q.index = fac_q.index.to_period('Q').to_timestamp('Q')

# Align GDP and factors
common_idx = gdp_q.index.intersection(fac_q.index)
gdp_a  = gdp_q.reindex(common_idx)
fac_a  = fac_q.reindex(common_idx)

# Train/test split
TRAIN_END   = pd.Timestamp('2020-12-31')
TEST_START  = pd.Timestamp('2021-01-01')
TEST_END    = pd.Timestamp('2023-06-30')

train_mask = gdp_a.index <= TRAIN_END
test_mask  = (gdp_a.index >= TEST_START) & (gdp_a.index <= TEST_END)

gdp_train = gdp_a[train_mask].dropna()
gdp_test  = gdp_a[test_mask].dropna()
fac_train = fac_a[train_mask].reindex(gdp_train.index)
fac_test  = fac_a[test_mask].reindex(gdp_test.index)

# ── AR(1) baseline ───────────────────────────────────────────────────────────
def ar1_predict(y_train, y_test):
    X_tr = sm.add_constant(y_train.shift(1).dropna())
    y_tr = y_train.reindex(X_tr.index)
    ols  = sm.OLS(y_tr, X_tr).fit()
    pred_in  = ols.predict(X_tr)
    # out-of-sample: one-step recursive
    X_te = sm.add_constant(y_test.shift(1).fillna(y_train.iloc[-1]))
    pred_out = ols.predict(X_te)
    return pred_in, pred_out, ols

pred_ar1_in, pred_ar1_out, ar1_fit = ar1_predict(gdp_train, gdp_test)

# ── AR(1) + Ft ───────────────────────────────────────────────────────────────
def ar1f_predict(y_train, y_test, f_train, f_test):
    X_tr = pd.concat([y_train.shift(1), f_train], axis=1).dropna()
    X_tr = sm.add_constant(X_tr)
    y_tr = y_train.reindex(X_tr.index)
    ols  = sm.OLS(y_tr, X_tr).fit()
    pred_in = ols.predict(X_tr)
    X_te = pd.concat([y_test.shift(1).fillna(y_train.iloc[-1]), f_test], axis=1)
    X_te = sm.add_constant(X_te)
    pred_out = ols.predict(X_te)
    return pred_in, pred_out, ols

pred_ar1f_in, pred_ar1f_out, ar1f_fit = ar1f_predict(
    gdp_train, gdp_test, fac_train, fac_test)

# ── RMSE ─────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0: return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

rmse_ar1_in   = rmse(gdp_train.reindex(pred_ar1_in.index), pred_ar1_in)
rmse_ar1_out  = rmse(gdp_test.reindex(pred_ar1_out.index), pred_ar1_out)
rmse_ar1f_in  = rmse(gdp_train.reindex(pred_ar1f_in.index), pred_ar1f_in)
rmse_ar1f_out = rmse(gdp_test.reindex(pred_ar1f_out.index), pred_ar1f_out)
improve_in    = (rmse_ar1_in - rmse_ar1f_in) / rmse_ar1_in * 100 if rmse_ar1_in else np.nan
improve_out   = (rmse_ar1_out - rmse_ar1f_out) / rmse_ar1_out * 100 if rmse_ar1_out else np.nan

print(f"  AR(1) RMSE    in={rmse_ar1_in:.4f}  out={rmse_ar1_out:.4f}")
print(f"  AR(1)+Ft RMSE in={rmse_ar1f_in:.4f}  out={rmse_ar1f_out:.4f}")
print(f"  Improvement   in={improve_in:.1f}%  out={improve_out:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — GDP nowcast | F1t | F2t  (3 subplots with vline)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 6...")
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

ax = axes[0]
ax.plot(gdp_a.index, gdp_a.values, color=COLORS['GDP'], lw=1.5, label='GDP (actual)')
full_in  = pd.concat([pred_ar1_in,  pred_ar1_out])
full_ar1f= pd.concat([pred_ar1f_in, pred_ar1f_out])
ax.plot(pred_ar1_in.index, pred_ar1_in.values, color=COLORS['AR1'],
        lw=1.2, linestyle='--', label='AR(1)')
ax.plot(pred_ar1_out.index, pred_ar1_out.values, color=COLORS['AR1'],
        lw=1.2, linestyle=':')
ax.plot(pred_ar1f_in.index, pred_ar1f_in.values, color=COLORS['AR1_F'],
        lw=1.2, label='AR(1) w/ $F_t$')
ax.plot(pred_ar1f_out.index, pred_ar1f_out.values, color=COLORS['AR1_F'],
        lw=1.2, linestyle=':')
ax.axvline(TEST_START, color=COLORS['VLINE'], lw=1, linestyle='--', alpha=0.8)
ax.text(TEST_START, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10,
        ' ← train | test →', fontsize=8, color=COLORS['VLINE'], va='top')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(loc='lower left', fontsize=8)
ax.set_title('Vietnam GDP Growth Nowcast (%YoY, Quarterly)')
ax.set_xlabel('Year')

ax = axes[1]
ax.plot(dates, scale_series(F1t_2), color=COLORS['F1'], lw=0.9)
ax.axvline(TEST_START, color=COLORS['VLINE'], lw=1, linestyle='--', alpha=0.6)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Scaled $F_{1t}$ (monthly, DFM $n=2$)')
ax.set_xlabel('Year')
fmt_date_axis(ax, dates)

ax = axes[2]
ax.plot(dates, scale_series(F2t_2), color=COLORS['F2'], lw=0.9)
ax.axvline(TEST_START, color=COLORS['VLINE'], lw=1, linestyle='--', alpha=0.6)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Scaled $F_{2t}$ (monthly, DFM $n=2$)')
ax.set_xlabel('Year')
fmt_date_axis(ax, dates)

plt.suptitle('GDP Nowcasting via DFM Factors — VN30/HOSE', y=1.01, fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure6.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.6 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Heatmap factor loadings (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 7...")
load_df = pd.DataFrame({'β₁ᵢ': load2[:, 0], 'β₂ᵢ': load2[:, 1]}, index=tickers)
load_df_sorted = load_df.sort_values('β₁ᵢ', ascending=False)

fig, ax = plt.subplots(figsize=(4, max(8, len(tickers) * 0.35)))
sns.heatmap(load_df_sorted, ax=ax, cmap=COLORS['HEATMAP'], center=0,
            annot=True, fmt='.3f', annot_kws={'size': 7},
            linewidths=0.3, cbar_kws={'shrink': 0.5, 'label': 'Loading value'})
ax.set_title('Factor Loadings $\\beta_{1i}$, $\\beta_{2i}$ — VN30/HOSE\n(sorted by $\\beta_{1i}$)')
ax.set_xlabel('Factor')
ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure7.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.7 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — VN-Index price with event annotations (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 8...")
vnindex_price = vnindex_close.dropna()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(vnindex_price.index, vnindex_price.values, color=COLORS['F1'], lw=1.2)
events = {
    '2018-01-31': 'US-China\ntrade war',
    '2020-03-31': 'COVID-19\nlockdown',
    '2022-03-31': 'Fed rate\nhike cycle',
    '2022-11-30': 'VN bond\nmarket crisis',
}
price_max = vnindex_price.max()
price_min = vnindex_price.min()
for date_str, label in events.items():
    dt = pd.Timestamp(date_str)
    ax.axvline(dt, color=COLORS['VLINE'], lw=1.2, linestyle='--', alpha=0.7)
    y_pos = price_max - 0.05 * (price_max - price_min)
    ax.text(dt, y_pos, label, fontsize=7, ha='center', va='top',
            color=COLORS['VLINE'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=COLORS['VLINE']))
ax.set_xlabel('Year')
ax.set_ylabel('VN-Index (points)')
ax.set_title('VN-Index with Key Market Events (2015–2023)')
fmt_date_axis(ax, vnindex_price.index)
ax.fill_between(vnindex_price.index, vnindex_price.values, price_min,
                color=COLORS['F1'], alpha=0.08)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure8.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.8 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — β_i / σ_i ratio bar chart (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 9...")
ratio = beta1 / np.where(sigma1 > 0, sigma1, np.nan)
ratio_df = pd.DataFrame({'ratio': ratio, 'ticker': tickers}).dropna()
ratio_df = ratio_df.sort_values('ratio', ascending=True).reset_index(drop=True)
med = ratio_df['ratio'].median()

bar_colors = [COLORS['BAR_POS'] if r >= med else COLORS['BAR_NEG']
              for r in ratio_df['ratio']]
fig, ax = plt.subplots(figsize=(5, max(7, len(ratio_df) * 0.32)))
bars = ax.barh(ratio_df['ticker'], ratio_df['ratio'], color=bar_colors,
               edgecolor='none', height=0.7)
ax.axvline(med, color='gray', lw=1, linestyle='--', alpha=0.7,
           label=f'Median = {med:.3f}')
ax.set_xlabel(r'$\beta_i / \sigma_i$ ratio')
ax.set_title('Market-driven vs Idiosyncratic Exposure\n'
             r'(High $\beta_i/\sigma_i$ = more market-driven)')
ax.legend(loc='lower right', fontsize=8)
ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure9.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.9 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Rolling 12-month correlation F1t vs VNIndex return (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Figure 10...")
window = 12
f1_s = pd.Series(scale_series(F1t), index=dates)
idx_s = vnindex_ret.reindex(dates).fillna(0)
rolling_corr = []
for i in range(len(dates)):
    start_i = max(0, i - window + 1)
    sl1 = f1_s.iloc[start_i:i+1]
    sl2 = idx_s.iloc[start_i:i+1]
    if len(sl1) >= 3:
        rolling_corr.append(np.corrcoef(sl1, sl2)[0, 1])
    else:
        rolling_corr.append(np.nan)
rolling_corr = pd.Series(rolling_corr, index=dates)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(dates, rolling_corr, color=COLORS['F1'], lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.fill_between(dates, rolling_corr, 0, where=(rolling_corr > 0),
                color=COLORS['F1'], alpha=0.15, label='Positive')
ax.fill_between(dates, rolling_corr, 0, where=(rolling_corr < 0),
                color=COLORS['VLINE'], alpha=0.15, label='Negative')
ax.set_ylabel('Rolling Correlation')
ax.set_xlabel('Year')
ax.set_title(f'Rolling {window}-month Correlation: $F_{{1t}}$ vs VN-Index Return')
ax.set_ylim(-1, 1)
ax.legend(loc='lower left', fontsize=8)
fmt_date_axis(ax, dates)
plt.tight_layout()
plt.savefig(f'{OUTFIG}/figure10.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Fig.10 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Correlation summary DFM n=1
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 1...")
corr_capm_b1  = np.corrcoef(beta1, capm_vals)[0, 1]
_, _, _, p_capm_b1, _ = stats.linregress(beta1, capm_vals)
corr_ft_psei  = np.corrcoef(F1t.values, vnindex_ret.fillna(0).values)[0, 1]
_, _, _, p_ft_psei, _ = stats.linregress(F1t.values, vnindex_ret.fillna(0).values)
corr_ft_pc1v  = np.corrcoef(F1t.values, PC1.values)[0, 1]
_, _, _, p_ft_pc1v, _ = stats.linregress(F1t.values, PC1.values)
corr_pc1_psei = np.corrcoef(PC1.values, vnindex_ret.fillna(0).values)[0, 1]
_, _, _, p_pc1_psei, _ = stats.linregress(PC1.values, vnindex_ret.fillna(0).values)

t1 = pd.DataFrame({
    'Series 1':   ['CAPM $\\hat{\\beta}_i$', '$F_t$', '$F_t$', 'PC1'],
    'Series 2':   ['$\\beta_i$', 'VN-Index', 'PC1', 'VN-Index'],
    'Correlation': [f'{corr_capm_b1:.4f}', f'{corr_ft_psei:.4f}',
                    f'{corr_ft_pc1v:.4f}', f'{corr_pc1_psei:.4f}'],
    'Sig.': [add_sig_stars(p_capm_b1), add_sig_stars(p_ft_psei),
             add_sig_stars(p_ft_pc1v), add_sig_stars(p_pc1_psei)],
})
t1.to_csv(f'{OUTTAB}/table1.csv', index=False)
with open(f'{OUTTAB}/table1.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Correlation Summary — DFM $n=1$, $p=3$ — VN30/HOSE}\n')
    f.write(t1.to_latex(index=False, escape=False))
    f.write('\\end{table}\n')
print("  Table 1 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — Loadings n=1
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 2...")
# T-stat for significance: β_i / se_β_i
# Estimate se using OLS of return on factor
beta_se1 = []
for i, col in enumerate(tickers):
    y = returns[col].values
    X = sm.add_constant(F1t.values)
    try:
        ols_res = sm.OLS(y, X).fit()
        beta_se1.append(ols_res.bse[1])
    except:
        beta_se1.append(np.nan)
beta_se1 = np.array(beta_se1)
beta_tstat1 = beta1 / np.where(beta_se1 > 0, beta_se1, np.nan)
beta_pval1  = 2 * (1 - stats.t.cdf(np.abs(beta_tstat1), df=T - 2))

load1_df = pd.DataFrame({
    'ticker': tickers,
    'beta_i': beta1,
    'se_beta': beta_se1,
    'sigma_i': sigma1,
    'sig': [add_sig_stars(p) for p in beta_pval1],
})
load1_df.to_csv(f'{OUTTAB}/table2.csv', index=False)
with open(f'{OUTTAB}/table2.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Factor Loadings — DFM $n=1$, $p=3$ — VN30/HOSE}\n')
    f.write('\\begin{tabular}{lrrrr}\\toprule\n')
    f.write('Stock & $\\beta_i$ & SE($\\beta_i$) & $\\sigma_i$ & Sig.\\\\\n\\midrule\n')
    for _, row in load1_df.iterrows():
        f.write(f'{row.ticker} & {row.beta_i:.4f} & {row.se_beta:.4f} & {row.sigma_i:.4f} & {row.sig} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
    f.write('\\footnotesize{***$p<0.01$, **$p<0.05$, *$p<0.10$, ns: not significant}\n')
    f.write('\\end{table}\n')
print("  Table 2 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — Correlation summary DFM n=2
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 3...")
corr_b1_capm = np.corrcoef(b1, capm_vals)[0, 1]
corr_b2_capm = np.corrcoef(b2, capm_vals)[0, 1]
corr_f1_idx2v = np.corrcoef(F1t_2.values, vnindex_ret.fillna(0).values)[0, 1]
corr_f2_idx2v = np.corrcoef(F2t_2.values, vnindex_ret.fillna(0).values)[0, 1]
corr_f1_pc1v2 = np.corrcoef(F1t_2.values, PC1.values)[0, 1]
corr_f2_pc2v  = np.corrcoef(F2t_2.values, PC2.values)[0, 1]
corr_pc2_idx  = np.corrcoef(PC2.values, vnindex_ret.fillna(0).values)[0, 1]

def lr_p(x, y):
    _, _, _, p, _ = stats.linregress(x, y); return p

t3 = pd.DataFrame({
    'Series 1':   ['CAPM $\\hat{\\beta}_i$', '$F_{1t}$', '$F_{1t}$', 'PC1',
                   'CAPM $\\hat{\\beta}_i$', '$F_{2t}$', '$F_{2t}$', 'PC2'],
    'Series 2':   ['$\\beta_{1i}$', 'VN-Index', 'PC1', 'VN-Index',
                   '$\\beta_{2i}$', 'VN-Index', 'PC2', 'VN-Index'],
    'Correlation':[f'{corr_b1_capm:.4f}', f'{corr_f1_idx2v:.4f}', f'{corr_f1_pc1v2:.4f}', f'{corr_pc1_psei:.4f}',
                   f'{corr_b2_capm:.4f}', f'{corr_f2_idx2v:.4f}', f'{corr_f2_pc2v:.4f}', f'{corr_pc2_idx:.4f}'],
    'Sig.': [add_sig_stars(lr_p(b1, capm_vals)),
             add_sig_stars(lr_p(F1t_2.values, vnindex_ret.fillna(0).values)),
             add_sig_stars(lr_p(F1t_2.values, PC1.values)),
             add_sig_stars(lr_p(PC1.values, vnindex_ret.fillna(0).values)),
             add_sig_stars(lr_p(b2, capm_vals)),
             add_sig_stars(lr_p(F2t_2.values, vnindex_ret.fillna(0).values)),
             add_sig_stars(lr_p(F2t_2.values, PC2.values)),
             add_sig_stars(lr_p(PC2.values, vnindex_ret.fillna(0).values))],
    'Factor': ['F1','F1','F1','F1','F2','F2','F2','F2'],
})
t3.to_csv(f'{OUTTAB}/table3.csv', index=False)
with open(f'{OUTTAB}/table3.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Correlation Summary — DFM $n=2$, $p=2$ — VN30/HOSE}\n')
    f.write(t3[['Factor','Series 1','Series 2','Correlation','Sig.']].to_latex(index=False, escape=False))
    f.write('\\end{table}\n')
print("  Table 3 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — Loadings n=2
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 4...")
beta_se2a, beta_se2b = [], []
for i, col in enumerate(tickers):
    y = returns[col].values
    X = sm.add_constant(np.column_stack([F1t_2.values, F2t_2.values]))
    try:
        ols_r = sm.OLS(y, X).fit()
        beta_se2a.append(ols_r.bse[1])
        beta_se2b.append(ols_r.bse[2])
    except:
        beta_se2a.append(np.nan); beta_se2b.append(np.nan)

beta_tstat2a = b1 / np.where(np.array(beta_se2a) > 0, beta_se2a, np.nan)
beta_pval2a  = 2 * (1 - stats.t.cdf(np.abs(beta_tstat2a), df=T - 3))
beta_tstat2b = b2 / np.where(np.array(beta_se2b) > 0, beta_se2b, np.nan)
beta_pval2b  = 2 * (1 - stats.t.cdf(np.abs(beta_tstat2b), df=T - 3))

load2_df = pd.DataFrame({
    'ticker': tickers,
    'beta_1i': b1, 'se_beta1': beta_se2a, 'sig1': [add_sig_stars(p) for p in beta_pval2a],
    'beta_2i': b2, 'se_beta2': beta_se2b, 'sig2': [add_sig_stars(p) for p in beta_pval2b],
    'sigma_i': sigma2,
})
load2_df.to_csv(f'{OUTTAB}/table4.csv', index=False)
with open(f'{OUTTAB}/table4.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Factor Loadings — DFM $n=2$, $p=2$ — VN30/HOSE}\n')
    f.write('\\begin{tabular}{lrrlrrl r}\\toprule\n')
    f.write('Stock & $\\beta_{1i}$ & SE & Sig & $\\beta_{2i}$ & SE & Sig & $\\sigma_i$ \\\\\n\\midrule\n')
    for _, row in load2_df.iterrows():
        f.write(f'{row.ticker} & {row.beta_1i:.4f} & {row.se_beta1:.4f} & {row.sig1} & '
                f'{row.beta_2i:.4f} & {row.se_beta2:.4f} & {row.sig2} & {row.sigma_i:.4f} \\\\\n')
    f.write('\\bottomrule\\end{tabular}\n')
    f.write('\\footnotesize{***$p<0.01$, **$p<0.05$, *$p<0.10$, ns: not significant}\n')
    f.write('\\end{table}\n')
print("  Table 4 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5 — GDP Nowcast RMSE
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 5...")
t5 = pd.DataFrame({
    'Model':         ['AR(1)', 'AR(1) w/ $F_t$'],
    'In-Sample':     [f'{rmse_ar1_in:.4f}',  f'{rmse_ar1f_in:.4f}'],
    'Out-of-Sample': [f'{rmse_ar1_out:.4f}', f'{rmse_ar1f_out:.4f}'],
    'Improvement(%)': ['-', f'{improve_out:.1f}%'],
})
t5.to_csv(f'{OUTTAB}/table5.csv', index=False)
with open(f'{OUTTAB}/table5.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Vietnam GDP Growth Rate Nowcast RMSE}\n')
    f.write(f'\\footnotesize{{Train: 2015Q2–2020Q4, Test: 2021Q1–2023Q2}}\n')
    f.write(t5.to_latex(index=False, escape=False))
    f.write('\\end{table}\n')
print("  Table 5 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 6 — AR/VAR coefficients (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 6...")
rows6 = []
# DFM n=1 AR(3)
for i, lam in enumerate(ar1_params):
    rows6.append({'Model': 'DFM $n=1$ AR(3)',
                  'Parameter': f'$\\Lambda_{{{i+1}}}$',
                  'Estimate': f'{lam:.4f}', 'Interpretation': f'Lag {i+1} coefficient'})
# DFM n=2 VAR(2)
if var2_params is not None and var2_params.ndim == 3:
    for lag in range(var2_params.shape[0]):
        for row in range(var2_params.shape[1]):
            for col in range(var2_params.shape[2]):
                rows6.append({'Model': 'DFM $n=2$ VAR(2)',
                              'Parameter': f'$\\Lambda_{{{row+1}{col+1},{lag+1}}}$',
                              'Estimate': f'{var2_params[lag, row, col]:.4f}',
                              'Interpretation': f'Lag {lag+1}, F{row+1}→F{col+1}'})

t6 = pd.DataFrame(rows6)
t6.to_csv(f'{OUTTAB}/table6.csv', index=False)
with open(f'{OUTTAB}/table6.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\n')
    f.write('\\caption{Factor Evolution Parameters — AR(3) for $n=1$, VAR(2) for $n=2$}\n')
    f.write(t6.to_latex(index=False, escape=False))
    f.write('\\end{table}\n')
print("  Table 6 done")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 7 — Descriptive statistics (NEW)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving Table 7...")
desc = returns.describe().T.round(4)
desc['skewness'] = returns.skew().round(4)
desc['kurtosis'] = returns.kurtosis().round(4)
desc = desc[['mean', 'std', 'skewness', 'kurtosis', 'min', 'max']]
desc.columns = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max']
desc.index.name = 'Stock'
desc.to_csv(f'{OUTTAB}/table7.csv')
with open(f'{OUTTAB}/table7.tex', 'w') as f:
    f.write('\\begin{table}[h]\\centering\\small\n')
    f.write('\\caption{Descriptive Statistics of VN30 Monthly Returns (\%) — 2015–2023}\n')
    f.write(desc.to_latex(float_format='%.4f', escape=False))
    f.write('\\end{table}\n')
print("  Table 7 done")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
with open('output/summary_results.txt', 'w') as f:
    f.write('=== DYNAMIC FACTOR ANALYSIS — VN30/HOSE ===\n\n')
    f.write(f'Sample period : {dates[0].date()} to {dates[-1].date()}\n')
    f.write(f'Stocks (S)    : {S}\n')
    f.write(f'Observations  : {T} monthly\n\n')
    f.write('--- DFM n=1 (factor_lag=0, factor_order=3, error_order=0) ---\n')
    f.write(f'Corr(Ft, VN-Index)   : {corr_ft_psei:.4f}  {add_sig_stars(p_ft_psei)}\n')
    f.write(f'Corr(Ft, PC1)        : {corr_ft_pc1v:.4f}  {add_sig_stars(p_ft_pc1v)}\n')
    f.write(f'Corr(beta_i, CAPM)   : {corr_capm_b1:.4f}  {add_sig_stars(p_capm_b1)}\n\n')
    f.write('--- DFM n=2 (factor_lag=0, factor_order=2, error_order=0) ---\n')
    f.write(f'Corr(F1t, VN-Index)  : {corr_f1_idx2v:.4f}\n')
    f.write(f'Corr(F2t, VN-Index)  : {corr_f2_idx2v:.4f}\n')
    f.write(f'Corr(beta1i, CAPM)   : {corr_b1_capm:.4f}\n')
    f.write(f'Corr(beta2i, CAPM)   : {corr_b2_capm:.4f}\n\n')
    f.write('--- GDP Nowcasting ---\n')
    f.write(f'AR(1) RMSE in-sample : {rmse_ar1_in:.4f}\n')
    f.write(f'AR(1) RMSE out-sample: {rmse_ar1_out:.4f}\n')
    f.write(f'AR(1)+Ft RMSE in     : {rmse_ar1f_in:.4f}\n')
    f.write(f'AR(1)+Ft RMSE out    : {rmse_ar1f_out:.4f}\n')
    f.write(f'Improvement in       : {improve_in:.2f}%\n')
    f.write(f'Improvement out      : {improve_out:.2f}%\n')

print('\n' + '=' * 60)
print('DONE — Output:')
print(f'  Figures 1-10 : {OUTFIG}/')
print(f'  Tables  1-7  : {OUTTAB}/  (CSV + LaTeX)')
print(f'  Summary      : output/summary_results.txt')
