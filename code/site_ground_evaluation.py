"""Ground-level litterfall and inventory evaluation.

"""

# In[] Imports
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# In[] Workflow
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ground_validation")
args = parser.parse_args()
Data_Dir = ROOT / "data/validation/ground"
Figure_Dir = args.output_dir.resolve()
if Figure_Dir == ROOT / "data" or ROOT / "data" in Figure_Dir.parents:
    raise ValueError("Outputs must not be written inside data/")
if Figure_Dir.exists():
    raise FileExistsError("Choose a new --output-dir to preserve existing results")

PATH_LITTERFALL_POOL = os.path.join(Data_Dir, 'Litterfall_global_pool_Zscore.csv')
PATH_LITTERFALL_SITE = os.path.join(Data_Dir, 'Litterfall_site_r_values.csv')
PATH_PLOT_CSV        = os.path.join(Data_Dir, 'Plot_Satellite_Dec_Qc_Final.csv')
PATH_GEI_PATCH       = os.path.join(Data_Dir, 'GEI_patch_level_data.csv')
PATH_GEI_SITE        = os.path.join(Data_Dir, 'GEI_site_level_stats.csv')

print("Loading cleaned ground-phenology alignment data from local CSV files...")
df_litterfall_pool = pd.read_csv(PATH_LITTERFALL_POOL)
df_litterfall_site = pd.read_csv(PATH_LITTERFALL_SITE)
df_plots           = pd.read_csv(PATH_PLOT_CSV)
df_gei_patch       = pd.read_csv(PATH_GEI_PATCH)
df_gei_site        = pd.read_csv(PATH_GEI_SITE)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

box_props = dict(
    boxstyle='round,pad=0.30',
    facecolor='white',
    alpha=0.90,
    edgecolor='0.80',
    linewidth=0.7
)

# In[] Functions
def add_panel_label(ax, label, x=-0.14, y=1.04):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight='bold',
        ha='left',
        va='bottom',
        clip_on=False
    )

def final_axis_style(ax):
    sns.despine(ax=ax)
    ax.tick_params(
        axis='both',
        direction='out',
        length=2.8,
        width=0.75,
        pad=2
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

def plot_nature_boxplot(ax, data, color, xlabel, title, ylabel):
    data = pd.Series(data).dropna().values

    ax.boxplot(
        [data],
        patch_artist=True,
        widths=0.42,
        showfliers=False,
        boxprops=dict(
            facecolor=color,
            alpha=0.50,
            edgecolor=color,
            linewidth=0.9
        ),
        medianprops=dict(color='black', linewidth=1.1),
        whiskerprops=dict(color='black', linewidth=0.9),
        capprops=dict(color='black', linewidth=0.9)
    )

    rng = np.random.default_rng(42)
    jitter_x = rng.normal(1, 0.035, size=len(data))

    ax.scatter(
        jitter_x,
        data,
        color='black',
        alpha=0.45,
        s=9,
        zorder=3
    )

    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1])
    ax.set_xticklabels([xlabel], fontweight='bold')
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, fontweight='bold', pad=6)

def format_p(p):
    if not np.isfinite(p):
        return "$p$ = NA"
    if p <= 0:
        return "$p$ < 10$^{-308}$"
    exponent = int(np.floor(np.log10(p)))
    coefficient = p / (10.0 ** exponent)
    return rf'$p$ = {coefficient:.2f} $\times$ 10$^{{{exponent}}}$'


fig = plt.figure(figsize=(7.2, 5.6), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[1.0, 0.92],
    hspace=0.42
)

gs_top = outer[0].subgridspec(1, 2, wspace=0.32)
gs_bottom = outer[1].subgridspec(1, 3, wspace=0.32)

ax_a = fig.add_subplot(gs_top[0, 0])
ax_b = fig.add_subplot(gs_top[0, 1])
ax_c = fig.add_subplot(gs_bottom[0, 0])
ax_d = fig.add_subplot(gs_bottom[0, 1])
ax_e = fig.add_subplot(gs_bottom[0, 2])

ax_a.scatter(
    df_litterfall_pool['Sat_Z'],
    df_litterfall_pool['Ground_Z'],
    alpha=0.23,
    color='0.45',
    s=13,
    edgecolor='none',
    rasterized=True
)

slope_a, intercept_a, r_value_a, p_value_a, _ = stats.linregress(
    df_litterfall_pool['Sat_Z'],
    df_litterfall_pool['Ground_Z']
)

range_a = (-2.1, 3.0)
xline_a = np.linspace(range_a[0], range_a[1], 200)

ax_a.plot(
    xline_a,
    slope_a * xline_a + intercept_a,
    color='#A50F15',
    linewidth=1.8
)

rmse_a = np.sqrt(
    np.nanmean(
        (df_litterfall_pool['Sat_Z'] - df_litterfall_pool['Ground_Z']) ** 2
    )
)

formula_a = (
    f"$y$ = {slope_a:.2f}$x$"
    if abs(intercept_a) < 0.01
    else f"$y$ = {slope_a:.2f}$x$ {'+' if intercept_a >= 0 else '-'} {abs(intercept_a):.2f}"
)

ax_a.text(
    0.04, 0.94,
    f"{formula_a}\n$r$ = {r_value_a:.2f}\nRMSE = {rmse_a:.2f}\n{format_p(p_value_a)}",
    transform=ax_a.transAxes,
    fontsize=7.2,
    va='top',
    ha='left',
    linespacing=1.15,
    bbox=box_props
)

ax_a.set_xlim(range_a)
ax_a.set_ylim(range_a)
ax_a.set_xlabel('Standardized deciduousness (Z-score)', labelpad=2)
ax_a.set_ylabel('Standardized litterfall (Z-score)', labelpad=3)

ax_inset = ax_a.inset_axes([0.69, 0.13, 0.25, 0.33])
ax_inset.patch.set_alpha(0.88)

sns.boxplot(
    y=df_litterfall_site['R_Value'],
    ax=ax_inset,
    width=0.42,
    color='white',
    linewidth=0.8,
    fliersize=0,
    boxprops={'edgecolor': '0.25', 'linewidth': 0.8},
    whiskerprops={'color': '0.25', 'linewidth': 0.8},
    capprops={'color': '0.25', 'linewidth': 0.8},
    medianprops={'color': '#A50F15', 'linewidth': 1.2}
)

np.random.seed(42)
sns.stripplot(
    y=df_litterfall_site['R_Value'],
    ax=ax_inset,
    color='0.20',
    size=1.8,
    alpha=0.55,
    jitter=0.10
)

ax_inset.set_title(
    f'Site-level $r$ ($n$={len(df_litterfall_site)})',
    fontsize=6.2,
    pad=2
)
ax_inset.set_ylabel('$r$', fontsize=6.2, labelpad=1)
ax_inset.set_xlabel('')
ax_inset.set_xticks([])
ax_inset.set_ylim(0.4, 1.05)
ax_inset.tick_params(axis='y', labelsize=6.0, length=2, width=0.6, pad=1)
sns.despine(ax=ax_inset, bottom=True)

df_plots_clean = df_plots.dropna(
    subset=['Deciduous_Proportion', 'Sat_Dec_Median']
).copy()

x_col = 'Deciduous_Proportion'
y_col = 'Sat_Dec_Median'

x = df_plots_clean[x_col].to_numpy(dtype=float)
y = df_plots_clean[y_col].to_numpy(dtype=float)

slope_b, intercept_b, r_value_b, p_value_b, _ = stats.linregress(x, y)
y_pred_b = slope_b * x + intercept_b
residual_sd_b = np.sqrt(np.sum((y - y_pred_b) ** 2) / (len(x) - 2))

x_fit_b = np.linspace(x.min(), x.max(), 300)
y_fit_b = slope_b * x_fit_b + intercept_b
x_mean_b = np.mean(x)
Sxx_b = np.sum((x - x_mean_b) ** 2)
t_crit_b = stats.t.ppf(0.975, df=len(x) - 2)
se_mean_b = residual_sd_b * np.sqrt(
    1 / len(x) + (x_fit_b - x_mean_b) ** 2 / Sxx_b
)
confidence_lower_b = y_fit_b - t_crit_b * se_mean_b
confidence_upper_b = y_fit_b + t_crit_b * se_mean_b

plot_min_b = min(-0.05, x.min() - 0.02, y.min() - 0.02)
plot_max_b = max(x.max() + 0.02, y.max() + 0.02)
plot_min_b = np.floor(plot_min_b * 20) / 20
plot_max_b = np.ceil(plot_max_b * 20) / 20

confidence_band_b = ax_b.fill_between(
    x_fit_b, confidence_lower_b, confidence_upper_b,
    color='#A50F15', alpha=0.16, linewidth=0, zorder=1,
    label='95% confidence interval'
)
axis_reference_b = np.array([plot_min_b, plot_max_b])
one_to_one_b, = ax_b.plot(
    axis_reference_b, axis_reference_b,
    linestyle='--', color='0.40', linewidth=0.9,
    alpha=0.60, zorder=0, label='1:1 line'
)
raw_points_b = ax_b.scatter(
    x, y, color='0.55', edgecolor='black',
    linewidth=0.25, alpha=0.38, s=14,
    rasterized=True, zorder=3, label='Raw plot data'
)
linear_fit_b, = ax_b.plot(
    x_fit_b, y_fit_b, color='#A50F15', linewidth=1.8,
    zorder=4, label='Raw linear fit'
)
sign_b = '+' if intercept_b >= 0 else '-'
stats_text_b = (
    f'$y$ = {slope_b:.2f}$x$ {sign_b} {abs(intercept_b):.2f}\n'
    f'$r$ = {r_value_b:.2f}\n'
    f'Residual s.d. = {residual_sd_b:.3f}\n'
    f'{format_p(p_value_b)}'
)
ax_b.text(
    0.04, 0.95, stats_text_b, transform=ax_b.transAxes,
    fontsize=7.1, va='top', ha='left',
    linespacing=1.10, bbox=box_props, zorder=5
)
ax_b.set_xlim(plot_min_b, plot_max_b)
ax_b.set_ylim(plot_min_b, plot_max_b)
ax_b.set_aspect('equal', adjustable='box')
ticks_b = np.arange(
    max(0, np.ceil(plot_min_b / 0.2) * 0.2),
    np.floor(plot_max_b / 0.2) * 0.2 + 0.001, 0.20
)
ax_b.set_xticks(ticks_b)
ax_b.set_yticks(ticks_b)
ax_b.set_xlabel('Ground-derived deciduous species proportion', labelpad=2)
ax_b.set_ylabel('Satellite-derived \ndeciduousness amplitude', labelpad=3)
ax_b.grid(True, linestyle=':', color='0.85', linewidth=0.6, zorder=0)
ax_b.legend(
    [raw_points_b, linear_fit_b, confidence_band_b, one_to_one_b],
    ['Raw plot data', 'Raw linear fit', '95% confidence interval', '1:1 line'],
    loc='lower right', frameon=True, fontsize=6.2, handlelength=1.8,
    borderpad=0.35, labelspacing=0.25, framealpha=0.90, edgecolor='0.85'
)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

COLOR_SCATTER = '#0072B2'
COLOR_R_BOX = '#2A9D8F'
COLOR_RMSE_BOX = '#D55E00'

ax_c.scatter(
    df_gei_patch['GE_deciduousness_abundance'],
    df_gei_patch['S2_deciduousness_abundance'],
    alpha=0.45,
    s=5,
    color=COLOR_SCATTER,
    edgecolor='none',
    rasterized=True
)

range_c = (-0.03, 1.05)

ax_c.plot(
    range_c,
    range_c,
    color='0.25',
    linestyle='--',
    linewidth=0.9,
    alpha=0.75
)

all_r = np.corrcoef(
    df_gei_patch['GE_deciduousness_abundance'],
    df_gei_patch['S2_deciduousness_abundance']
)[0, 1]

all_rmse = np.sqrt(
    np.nanmean(
        (
            df_gei_patch['S2_deciduousness_abundance']
            - df_gei_patch['GE_deciduousness_abundance']
        ) ** 2
    )
)

ax_c.text(
    0.06, 0.94,
    f"$r$ = {all_r:.2f}\nRMSE = {all_rmse:.2f}",
    transform=ax_c.transAxes,
    fontsize=7.4,
    va='top',
    ha='left'
)

ax_c.set_xlim(range_c)
ax_c.set_ylim(range_c)
ax_c.set_xlabel('GEI derived deciduousness abundance', labelpad=2)
ax_c.set_ylabel('S2 derived deciduousness abundance', labelpad=3)

plot_nature_boxplot(
    ax=ax_d,
    data=df_gei_site['r_value'],
    color=COLOR_R_BOX,
    xlabel='$r$',
    title='GEI-specific $r$',
    ylabel='$r$'
)

r_values = pd.Series(df_gei_site['r_value']).dropna()
r_pad = 0.08 * (r_values.max() - r_values.min())
ax_d.set_ylim(r_values.min() - r_pad, r_values.max() + r_pad)

plot_nature_boxplot(
    ax=ax_e,
    data=df_gei_site['rmse_value'],
    color=COLOR_RMSE_BOX,
    xlabel='RMSE',
    title=' GEI-specific RMSE',
    ylabel='RMSE'
)

rmse_values = pd.Series(df_gei_site['rmse_value']).dropna()
ax_e.set_ylim(0, rmse_values.max() * 1.12)

for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], ['a', 'b', 'c', 'd', 'e']):
    final_axis_style(ax)
    add_panel_label(ax, label)

fig.subplots_adjust(
    left=0.085,
    right=0.950,
    bottom=0.08,
    top=0.93
)

Figure_Dir.mkdir(parents=True)
for extension in ("png", "pdf"):
    fig.savefig(Figure_Dir / f"Extended_Data_Fig3_Combined_Final.{extension}",
                dpi=600, bbox_inches='tight', pad_inches=0.03)
pd.DataFrame({
    "ground_proportion": x, "satellite_amplitude": y,
    "fitted_amplitude": y_pred_b,
}).to_csv(Figure_Dir / "inventory_raw_fit.csv", index=False)
pd.DataFrame({
    "ground_proportion": x_fit_b, "fitted_mean": y_fit_b,
    "mean_ci95_lower": confidence_lower_b, "mean_ci95_upper": confidence_upper_b,
}).to_csv(Figure_Dir / "inventory_confidence_band.csv", index=False)
report = {
    "scope": "Visualization from released processed validation CSVs, not raw-observation reconstruction.",
    "litterfall": {"n": len(df_litterfall_pool), "slope": slope_a,
                  "intercept": intercept_a, "r": r_value_a, "p": p_value_a,
                  "agreement_rmse": rmse_a},
    "inventory": {"n": len(x), "slope": slope_b, "intercept": intercept_b,
                  "r": r_value_b, "p": p_value_b, "residual_sd": residual_sd_b,
                  "interval": "95% confidence interval of the OLS fitted mean"},
    "gei": {"n": len(df_gei_patch), "r": all_r, "agreement_rmse": all_rmse},
}
(Figure_Dir / "ground_validation_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
plt.close(fig)
