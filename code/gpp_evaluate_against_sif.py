"""GPP evaluation against SIF and eddy-flux benchmarks.

"""

# In[] Imports
import os
import cv2
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from osgeo import gdal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
import warnings
import cartopy.io.shapereader as shpreader

# In[] Workflow
matplotlib.use("Agg")
warnings.filterwarnings('ignore')

COLOR_EXP3 = '#e34a33'
COLOR_EXP1 = '#1f77b4'
COLOR_OBS = 'k'

# In[] Functions
def readTif_gdal_safe(fileName, nbands=12):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)
    if dataset is None: return None, None, None
    im_width, im_height = dataset.RasterXSize, dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    if im_data.ndim == 3 and im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])
    im_data = im_data.astype(np.float32)

    im_data[im_data == 65535] = np.nan
    im_data[im_data == -9999] = np.nan
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def normalize_ts(x):
    x_min = np.nanmin(x, axis=1, keepdims=True)
    x_max = np.nanmax(x, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x - x_min) / (x_max - x_min)

def vectorized_pearsonr(x, y):
    joint_valid = ~np.isnan(x) & ~np.isnan(y)
    x_safe = np.where(joint_valid, x, np.nan)
    y_safe = np.where(joint_valid, y, np.nan)
    xm = x_safe - np.nanmean(x_safe, axis=1, keepdims=True)
    ym = y_safe - np.nanmean(y_safe, axis=1, keepdims=True)
    r_num = np.nansum(xm * ym, axis=1)
    r_den = np.sqrt(np.nansum(xm ** 2, axis=1) * np.nansum(ym ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return r_num / r_den

def add_shp(ax, shp_path, **kwargs):
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(shp_path)
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()

print(">>> Initializing masks and SIF validation data...")
raws_y, columns_x = 786, 650
img_extent = [-79.77, -44.51, -20.52, 8.62]
map_extent = [-80, -44, -22, 10]
shp_path = r'data/boundaries/Amazon_ThreeRegions_Clip.shp'

cls_path = r'data/forest_mask/MCD12Q1_Amazon.tif'
_, _, cls_md = readTif_gdal_safe(cls_path)
cls_md_forest = cls_md.copy()
cls_md_forest[cls_md_forest != 2] = -1
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x))
forest_mask = (cls_modis_coarse == 2)

sif_path_1 = r'data/gpp/sif/Amazon_GOSIF_mean_calibrated.tif'
sif_path_2 = r'data/gpp/sif/CSIF_Amazon_005_calibrated.tif'
_, _, sif_1 = readTif_gdal_safe(sif_path_1)
_, _, sif_2 = readTif_gdal_safe(sif_path_2)

sif_1 = cv2.resize(sif_1, (raws_y, columns_x))
sif_2 = cv2.resize(sif_2, (raws_y, columns_x))

sif1_ts = sif_1[forest_mask, :] * 0.0001

sif2_ts = sif_2[forest_mask, :]
sif1_ts[sif1_ts < 0] = np.nan
sif2_ts[sif2_ts < 0] = np.nan
sif_ref_nor = np.nanmean(np.stack([normalize_ts(sif1_ts), normalize_ts(sif2_ts)], axis=0), axis=0)

print(">>> Loading Exp1 and Exp3 outputs for comparison...")

exp1_dir = r'data/gpp/outputs'
exp3_dir = r'data/gpp/outputs'

model_pairs = {
    'EC-LUE': (['EC_LUE_EVI_GPP_local.tif', 'EC_LUE_kNDVI_GPP_local.tif', 'EC_LUE_NDVI_GPP_local.tif',
                'EC_LUE_MODIS_LAI_GPP_local.tif'],
               'EC_LUE_LAI_Dec_Demography_GPP.tif'),
    'MODIS-LUE': (['MOD_LUE_GPP_local.tif'],
                'MOD_LUE_LAI_Dec_Demography_GPP.tif'),
    'TL-EC': (['TL_EC_LUE_GPP_local.tif'],
              'TL_EC_LUE_LAI_Dec_Demography_GPP.tif')
}

delta_r_maps = {}
formulation_mean_delta_r_list = []

for model_name, (exp1_files, file_exp3) in model_pairs.items():

    exp3_path = os.path.join(exp3_dir, file_exp3)
    if not os.path.exists(exp3_path):
        print(f"Missing Exp3 file: {file_exp3}; skipping this model.")
        continue

    geo_info, _, data_exp3 = readTif_gdal_safe(exp3_path)
    data_exp3 = cv2.resize(data_exp3, (raws_y, columns_x))
    ts_exp3 = data_exp3[forest_mask, :]

    specific_delta_r_maps = []

    for f1 in exp1_files:
        f_path = os.path.join(exp1_dir, f1)
        if not os.path.exists(f_path):
            print(f"Missing file: {f1}; skipping this file.")
            continue

        _, _, data_exp1 = readTif_gdal_safe(f_path)
        data_exp1 = cv2.resize(data_exp1, (raws_y, columns_x))
        ts_exp1 = data_exp1[forest_mask, :]

        valid_mask_ind = (np.sum(~np.isnan(sif_ref_nor), axis=1) >= 6) & \
                         (np.sum(~np.isnan(ts_exp1), axis=1) >= 6) & \
                         (np.sum(~np.isnan(ts_exp3), axis=1) >= 6)

        r_exp1 = vectorized_pearsonr(ts_exp1[valid_mask_ind], sif_ref_nor[valid_mask_ind])
        r_exp3 = vectorized_pearsonr(ts_exp3[valid_mask_ind], sif_ref_nor[valid_mask_ind])

        delta_r_ind = r_exp3 - r_exp1

        map_delta_r_ind = np.full(forest_mask.shape, np.nan)
        temp_r_ind = np.full(forest_mask.sum(), np.nan)
        temp_r_ind[valid_mask_ind] = delta_r_ind
        map_delta_r_ind[forest_mask] = temp_r_ind

        specific_delta_r_maps.append(map_delta_r_ind)

    if specific_delta_r_maps:

        mean_formulation_delta_r = np.nanmean(np.stack(specific_delta_r_maps, axis=0), axis=0)

        delta_r_maps[model_name] = mean_formulation_delta_r
        formulation_mean_delta_r_list.append(mean_formulation_delta_r)

        print(f"Finished comparison: {model_name} using {len(specific_delta_r_maps)} specific-model Delta r maps.")
    else:
        print(f"No valid specific models were available for {model_name}.")

print(">>> Calculating ensemble mean improvement...")
if formulation_mean_delta_r_list:

    ensemble_mean_delta_r = np.nanmean(np.stack(formulation_mean_delta_r_list, axis=0), axis=0)
    delta_r_maps['Ensemble Mean'] = ensemble_mean_delta_r
    print("Finished ensemble mean calculation.")
else:
    print("Not enough formulation data to calculate the ensemble mean.")

print(">>> Extracting site-scale time series and +/-1 SEM bands...")

eddy_flux_name_list = ['K34_CfluxBF_DOY.csv', 'K67_CfluxBF_DOY.csv', 'CAX_CfluxBF_DOY.csv', 'RJA_CfluxBF_DOY.csv']
site_locate_path = r'data/validation/eddy_flux/Eddy_Fluxes_Sites_Locations.xlsx'
eddy_flux_dir = r'data/validation/eddy_flux'
site_locate_pd = pd.read_excel(site_locate_path)

site_series_data = {}
r_bench_all, rmse_bench_all = [], []
r_ens_all, rmse_ens_all = [], []
EC_pos = []

raw_exp1_matrices = []
raw_exp3_matrices = []
raw_exp1_labels = []
raw_exp3_labels = []

for model_name, (exp1_files, file_exp3) in model_pairs.items():
    formulation_exp1_matrices = []

    for f1 in exp1_files:
        f_path = os.path.join(exp1_dir, f1)
        if os.path.exists(f_path):
            _, _, data = readTif_gdal_safe(f_path)
            formulation_exp1_matrices.append(cv2.resize(data, (raws_y, columns_x)))

    if len(formulation_exp1_matrices) > 0:
        raw_exp1_matrices.append(np.nanmean(np.stack(formulation_exp1_matrices, axis=0), axis=0))
        raw_exp1_labels.append(model_name)

    exp3_path = os.path.join(exp3_dir, file_exp3)
    if os.path.exists(exp3_path):
        _, _, data_exp3 = readTif_gdal_safe(exp3_path)
        raw_exp3_matrices.append(cv2.resize(data_exp3, (raws_y, columns_x)))
        raw_exp3_labels.append(model_name)

print(
    f"Site extraction uses formulation-balanced Exp1 models: {raw_exp1_labels}; "
    f"LD(Exp3) models: {raw_exp3_labels}.")

for eddy_flux_name in eddy_flux_name_list:
    site_name = eddy_flux_name.split('_')[0]
    site_info = site_locate_pd[site_locate_pd['Site'] == site_name]
    lat, lon = site_info['Latitude'].values[0], site_info['Longitude'].values[0]
    EC_pos.append((lat, lon))

    col = int((lon - geo_info[0]) / geo_info[1])
    row = int((lat - geo_info[3]) / geo_info[5])

    r_start, r_end = max(0, row - 1), row + 1
    c_start, c_end = max(0, col - 1), col + 1

    site_exp3_models = np.array([
        np.nanmean(d[r_start:r_end, c_start:c_end, :], axis=(0, 1))
        for d in raw_exp3_matrices
    ])
    site_exp3_mean = np.nanmean(site_exp3_models, axis=0)

    site_exp3_sem = np.nanstd(site_exp3_models, axis=0, ddof=1) / np.sqrt(site_exp3_models.shape[0])

    site_exp1_models = np.array([
        np.nanmean(d[r_start:r_end, c_start:c_end, :], axis=(0, 1))
        for d in raw_exp1_matrices
    ])
    site_exp1_mean = np.nanmean(site_exp1_models, axis=0)
    site_exp1_sem = np.nanstd(site_exp1_models, axis=0, ddof=1) / np.sqrt(site_exp1_models.shape[0])

    csv_path = os.path.join(eddy_flux_dir, eddy_flux_name)
    pd_eddy = pd.read_csv(csv_path, usecols=['DOY', 'GEP', 'Month'])
    pd_eddy['GEP'] = pd.to_numeric(pd_eddy['GEP'], errors='coerce')
    site_obs = pd_eddy.groupby('Month')['GEP'].mean().to_numpy()

    v_mask = ~np.isnan(site_obs) & ~np.isnan(site_exp3_mean) & ~np.isnan(site_exp1_mean)

    if np.sum(v_mask) > 3:

        r_exp3 = np.corrcoef(site_exp3_mean[v_mask], site_obs[v_mask])[0, 1]
        rmse_exp3 = np.sqrt(np.mean((site_exp3_mean[v_mask] - site_obs[v_mask]) ** 2))
        r_bench_all.append(r_exp3)
        rmse_bench_all.append(rmse_exp3)

        r_exp1 = np.corrcoef(site_exp1_mean[v_mask], site_obs[v_mask])[0, 1]
        rmse_exp1 = np.sqrt(np.mean((site_exp1_mean[v_mask] - site_obs[v_mask]) ** 2))
        r_ens_all.append(r_exp1)
        rmse_ens_all.append(rmse_exp1)
    else:
        print(f"Not enough valid months at site {site_name}; R and RMSE were not calculated.")

    site_series_data[site_name] = {
        'obs': site_obs,
        'exp3_m': site_exp3_mean, 'exp3_sem': site_exp3_sem,
        'exp1_m': site_exp1_mean, 'exp1_sem': site_exp1_sem
    }

print("Finished extracting all site time series and error bands.")

cmap_blue = plt.cm.Blues_r(np.linspace(0.10, 0.80, 5))
cmap_red = plt.cm.Reds(np.linspace(0.10, 0.80, 5))
colors = np.vstack((cmap_blue, cmap_red))
custom_cmap = mcolors.ListedColormap(colors)

levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=10)

print(">>> Generating main Figure 4...")
import string

fig_main = plt.figure(figsize=(15.5,8), dpi=150)
proj = ccrs.PlateCarree()

plt.rcParams['font.family'] = 'Helvetica'

ax_map = fig_main.add_axes([0.05, 0.20, 0.46, 0.75], projection=proj)

ax_map.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax_map.add_feature(cfeature.OCEAN, zorder=2)

add_shp(ax_map, shp_path, lw=0.8, ec='k', fc='none', zorder=4)

map_data_ens = delta_r_maps['Ensemble Mean']
im_main = ax_map.imshow(map_data_ens, origin='upper', extent=img_extent, transform=proj,
                        zorder=3, cmap=custom_cmap, norm=norm)
ax_map.set_extent(map_extent, crs=proj)
ax_map.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax_map.set_yticks(np.arange(-20, 20, 10), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter())
ax_map.yaxis.set_major_formatter(LatitudeFormatter())

ax_map.tick_params(axis='both', labelsize=16, colors='k', length=4, width=0.8, direction='out',
                   left=True, right=False, top=False, bottom=True,
                   labelright=False, labeltop=False, labelbottom=True)

ax_map.text(-0.05, 1.02, 'a', transform=ax_map.transAxes, fontsize=18, fontweight='bold', va='bottom')

for i, pos in enumerate(EC_pos):
    ax_map.plot(pos[1], pos[0], marker='*', color='k', ms=20, mec='white', zorder=5)
    ax_map.text(pos[1] - 0.6, pos[0] - 2.5, eddy_flux_name_list[i].split('_')[0], color="k", fontsize=14,
                fontweight='bold', zorder=6)

cbar_ax = fig_main.add_axes([0.10, 0.11, 0.36, 0.025])
cb_main = fig_main.colorbar(im_main, cax=cbar_ax, orientation='horizontal')
cb_main.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])

cb_main.set_ticklabels(['<= -0.8', '-0.4', '0', '0.4', r'0.8 <='], fontsize=14)
cb_main.set_label(r'$\Delta r$', fontsize=16, fontweight='bold')

cb_main.ax.tick_params(direction='in', length=5, width=1.2, colors='k')
cb_main.outline.set_linewidth(1.2)

valid_data = map_data_ens[~np.isnan(map_data_ens)]
if len(valid_data) > 0:
    improved_cnt = np.sum(valid_data >= 0)
    degraded_cnt = np.sum(valid_data < 0)
    sizes = [improved_cnt, degraded_cnt]
    pie_colors_sig = ['#cb181d', '#2166ac']

    ax_ins = ax_map.inset_axes([0.70, 0.01, 0.34, 0.34], zorder=10)
    wedges, texts, autotexts = ax_ins.pie(
        sizes, colors=pie_colors_sig, autopct='%1.1f%%', startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor=None, linewidth=0.5, alpha=0.9, width=0.65)
    )

    for j, autotext in enumerate(autotexts):
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
        if sizes[j] / sum(sizes) < 0.05:
            autotext.set_text('')

line_coords = [[0.57, 0.71, 0.18, 0.22], [0.80, 0.71, 0.18, 0.22], [0.57, 0.395, 0.18, 0.22], [0.80, 0.395, 0.18, 0.22]]
m_list = np.arange(1, 13)
line_handles = []

for i, site in enumerate([n.split('_')[0] for n in eddy_flux_name_list]):
    ax = fig_main.add_axes(line_coords[i])
    d = site_series_data[site]

    ax.fill_between(m_list, d['exp3_m'] - d['exp3_sem'], d['exp3_m'] + d['exp3_sem'], color=COLOR_EXP3, alpha=0.20,
                    lw=0, zorder=2)
    l1 = ax.plot(m_list, d['exp3_m'], color=COLOR_EXP3, ls='-', lw=2, marker='^', ms=7, mfc=COLOR_EXP3, mec='white',
                 zorder=5)

    ax.fill_between(m_list, d['exp1_m'] - d['exp1_sem'], d['exp1_m'] + d['exp1_sem'], color=COLOR_EXP1, alpha=0.20,
                    lw=0, zorder=1)
    l2 = ax.plot(m_list, d['exp1_m'], color=COLOR_EXP1, ls='-', lw=2, marker='s', ms=6, mfc=COLOR_EXP1, mec='white',
                 zorder=4)

    l3 = ax.plot(m_list, d['obs'], color=COLOR_OBS, ls='--', lw=1.5, marker='o', ms=6, mfc='white', mec=COLOR_OBS,
                 zorder=6)
    if i == 0: line_handles = [l1[0], l2[0], l3[0]]

    ax.tick_params(axis='both', direction='out', labelsize=13)

    ax.set_title(f'{site}', fontsize=15, fontweight='bold', pad=8)

    letter = string.ascii_lowercase[i + 1]
    ax.text(0.00, 1.05, letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

    if i % 2 == 0: ax.set_ylabel(r'GPP $(\mathrm{g\,C\,m^{-2}\,d^{-1}})$', fontsize=15)
    ax.set_xticks(m_list)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=13)
    ax.set_ylim(4.5, 11.0 if site != 'RJA' else 11.5)
    ax.set_xlim(0.5, 12.5)

fig_main.legend(handles=line_handles, labels=['LD-LUE +/- 1 SEM', 'Conv-LUE +/- 1 SEM', 'EC-derived GPP'],
                loc='center', bbox_to_anchor=(0.775, 0.325), ncol=3, frameon=False, fontsize=13)

bar_y = np.arange(4)
width = 0.35

ax_r = fig_main.add_axes([0.57, 0.12, 0.18, 0.15])
b1 = ax_r.barh(bar_y - width / 2, r_bench_all[::-1], width, color=COLOR_EXP3, zorder=3)
b2 = ax_r.barh(bar_y + width / 2, r_ens_all[::-1], width, color=COLOR_EXP1, zorder=3)

ax_r.tick_params(axis='both', direction='out', labelsize=13)

ax_r.set_title('', loc='left')
ax_r.text(0.00, 1.05, 'f', transform=ax_r.transAxes, fontsize=18, fontweight='bold', va='bottom')

ax_r.set_xlabel('Correlation coefficient ($r$)', fontsize=15)
ax_r.set_yticks(bar_y)
ax_r.set_yticklabels([n.split('_')[0] for n in eddy_flux_name_list][::-1])

ax_rmse = fig_main.add_axes([0.80, 0.12, 0.18, 0.15])
ax_rmse.barh(bar_y - width / 2, rmse_bench_all[::-1], width, color=COLOR_EXP3, zorder=3)
ax_rmse.barh(bar_y + width / 2, rmse_ens_all[::-1], width, color=COLOR_EXP1, zorder=3)

ax_rmse.tick_params(axis='both', direction='out', labelsize=13)

ax_rmse.set_title('', loc='left')
ax_rmse.text(0.00, 1.05, 'g', transform=ax_rmse.transAxes, fontsize=18, fontweight='bold', va='bottom')

ax_rmse.set_xlabel(r'RMSE $(\mathrm{g\,C\,m^{-2}\,d^{-1}})$', fontsize=15)
ax_rmse.set_yticks(bar_y)
ax_rmse.set_yticklabels([n.split('_')[0] for n in eddy_flux_name_list][::-1])

fig_main.legend(handles=[b1, b2], labels=['LD-LUE', 'Conv-LUE'],
                loc='center', bbox_to_anchor=(0.775, 0.025), ncol=2, frameon=False, fontsize=15)

plt.show()

save_path = r'outputs/figures/FigS_LD_LUE_Conv_LUE_Comparison.png'
fig_main.savefig(save_path, dpi=300, bbox_inches='tight')

print(">>> Generating the supplementary evaluation matrix...")
rows, cols = 1, 3
fig_sup = plt.figure(figsize=(12, 4.5), dpi=150)

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(rows, cols, figure=fig_sup, wspace=0.15)

plot_order = ['EC-LUE', 'TL-EC', 'MODIS-LUE']
plot_tile = ['EC-LUE', 'TL-EC-LUE', 'MODIS-LUE']

panel_letters = ['a', 'b', 'c']

pie_colors = ['#cb181d', '#2166ac', '#e0e0e0']

for i, model_name in enumerate(plot_order):
    ax = fig_sup.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
    map_data = delta_r_maps[model_name]

    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN,  edgecolor='none', zorder=2, alpha=0.8)

    add_shp(ax, shp_path, lw=0.8, ec='k', fc='none', zorder=5)

    im = ax.imshow(map_data, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(),
                   zorder=4, cmap=custom_cmap, norm=norm)

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-80, -40, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-20, 20, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    show_left_label = (i == 0)
    ax.tick_params(axis='both', labelsize=14, colors='k', length=4, width=0.8,
                   left=True, right=False, top=False, bottom=True,
                   labelleft=show_left_label, labelright=False, labeltop=False, labelbottom=True)

    model_name = plot_tile[i]
    ax.set_title(f'{model_name}', fontsize=16, pad=12, y=1.00)
    ax.text(-0.1, 1.08, panel_letters[i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

    valid_data = map_data[~np.isnan(map_data)]
    if len(valid_data) > 0:
        improved_cnt = np.sum(valid_data >= 0)
        degraded_cnt = np.sum(valid_data < 0)

        sizes = [improved_cnt, degraded_cnt]
        pie_colors_sig = ['#cb181d', '#2166ac']

        ax_ins = ax.inset_axes([0.72, 0.01, 0.34, 0.34], zorder=10)

        wedges, texts, autotexts = ax_ins.pie(
            sizes,
            colors=pie_colors_sig,
            autopct='%1.1f%%',
            startangle=90,

            pctdistance=0.65,

            wedgeprops=dict(edgecolor=None, linewidth=0.5, alpha=0.9, width=0.65)
        )

        for j, autotext in enumerate(autotexts):

            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
            autotext.set_color('k')

            if sizes[j] / sum(sizes) < 0.05:
                autotext.set_text('')

fig_sup.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.22)

cbar_ax = fig_sup.add_axes([0.25, 0.12, 0.5, 0.025])
cb_main = fig_sup.colorbar(im, cax=cbar_ax, orientation='horizontal')
cb_main.ax.xaxis.set_ticks(np.array([-0.80, -0.40, 0, 0.40, 0.80]))

cb_main.set_ticklabels(['<= -0.8', '-0.4', '0', '0.4', r'0.8 <='],fontsize=14)
cb_main.set_label(r'$\Delta r$', fontsize=16, fontweight='bold')

cb_main.outline.set_linewidth(0.8)

plt.show()
print("\n>>> Finished drawing all figures.")

save_path_sup = r'outputs/figures/FigS_Three_Framework_LD_LUE_Conv_LUE_Comparison_Supplement.png'
fig_sup.savefig(save_path_sup, dpi=300, bbox_inches='tight')
