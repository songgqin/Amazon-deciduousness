"""Figure 3 asynchrony driver analysis.

"""

# In[] Imports
from bayes_opt import BayesianOptimization
import seaborn as sns
from scipy import stats
from amazon_preprocessing import readTif_gdal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import os
import xgboost as xgb
import cv2
import matplotlib
from osgeo import gdal
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# In[] Workflow
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 200
import shap
import copy
from skimage import morphology
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import itertools

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True

matplotlib.rcParams['figure.dpi'] = 150

# In[] Functions
def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    datatype = gdal.GDT_Float32

    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype)

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)

    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(1).SetNoDataValue(np.nan)

    del outputData

def remove_outliers(datax, datay):

    df = pd.DataFrame({'datax': datax, 'datay': datay})

    group_stats = df.groupby('datax')['datay'].agg(['mean', 'std'])

    group_stats['lower_bound'] = group_stats['mean'] - 3 * group_stats['std']
    group_stats['upper_bound'] = group_stats['mean'] + 3 * group_stats['std']

    df = df.merge(group_stats, left_on='datax', right_index=True)

    mask = (df['datay'] >= df['lower_bound']) & (df['datay'] <= df['upper_bound'])
    df_filtered = df[mask]

    return df_filtered['datax'].values, df_filtered['datay'].values

raws_y, columns_x = 786, 650
cls_modis_path = r'data/forest_mask/MCD12Q1_Amazon.tif'

_, _, cls_md = readTif_gdal(cls_modis_path)
cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)
cls_md_forest[cls_md_forest != 2] = np.nan
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = ~np.isnan(cls_modis_coarse)

ind_where = np.where(forest_mask == 1)
print('evergreen forest pixels:', len(ind_where[0]))

variables_dir = r'data/drivers/inputs'

variable_lists = [
    'climate_par.tif',
    'hydroclimate_Amazon_MCWD_mean_3yr.tif',
    'hydroclimate_precipitation_ERA.tif',
    'hydroclimate_herbivory_Amazon.tif',
    'soil_clay.tif',
    'soil_fertility.tif',
    'soil_moisture_Amazon_3layers.tif',
    'soil_sand.tif',
]

variable_lists.sort()

test_variable_list = variable_lists
variables_all = np.zeros((650, 786, len(test_variable_list)))

save_dir = r'outputs'

for variable_name in test_variable_list:
    _geo, _prj, variable = readTif_gdal(os.path.join(variables_dir, variable_name))
    variable = variable.astype(np.float32)

    if 'precipitation' in variable_name:
        variable_mean = np.nansum(variable, axis=2) * 1000.0

    elif 'par' in variable_name:
        variable_mean = np.nanmean(variable, axis=2)
        variable_mean[variable_mean < 100] = np.nan
    elif 'temperature' in variable_name:
        variable_mean = np.nanmean(variable, axis=2) - 273.15
    elif 'vpd' in variable_name:

        variable_mean = variable

        variable_mean[variable_mean < 0] = np.nan

    elif 'wtd' in variable_name:
        variable_mean = -variable
        variable_mean[variable_mean > 1000] = np.nan
    elif 'sand' in variable_name or 'clay' in variable_name:
        variable_mean = np.nanmean(variable, axis=2)

    elif 'MCWD' in variable_name:
        variable_mean = copy.deepcopy(variable)
        variable_mean[variable_mean < -800] = np.nan

    else:
        variable_mean = variable

    variable_mean = cv2.resize(variable_mean, (786, 650), interpolation=cv2.INTER_LINEAR)

    variable_mean[~forest_mask] = np.nan

    variables_all[:, :, test_variable_list.index(variable_name)] = variable_mean

variable_reshape = variables_all.reshape(-1, variables_all.shape[2])

par, fertiltiy, vpd = variables_all[:, :, 1], variables_all[:, :, 6], variables_all[:, :, -1]

variable_name_list = [x.split('.')[0] for x in test_variable_list]

variable_reshape_vif = copy.deepcopy(variable_reshape.astype(np.float64))

inf_mask = np.any(np.isinf(variable_reshape_vif), axis=1)
nan_mask = np.any(np.isnan(variable_reshape_vif), axis=1)
final_mask = np.logical_or(inf_mask, nan_mask)

variable_reshape_vif = variable_reshape_vif[~final_mask]

pd_data = pd.DataFrame(variable_reshape_vif, columns=variable_name_list)

vif_data = add_constant(pd_data)
vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])],
                index=vif_data.columns)

deciduousness_path = r'data/seasonality/time_lag_map_0521.tif'

_geo, _prj, deciduousness = readTif_gdal(deciduousness_path)

annual_deciduousness = copy.deepcopy(deciduousness)

annual_deciduousness[~forest_mask] = np.nan
annual_deciduousness = cv2.medianBlur(annual_deciduousness.astype(np.float32), 3)

deciduousness_res = annual_deciduousness.reshape(-1)

inf_mask = np.any(np.isinf(variable_reshape), axis=1)
nan_mask = np.logical_or(np.isnan(deciduousness_res), np.any(np.isnan(variable_reshape), axis=1))

final_mask = np.logical_or(inf_mask, nan_mask)

data_y = deciduousness_res[~final_mask]

data_x = variable_reshape[~final_mask]

dec_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'

dec_geo, dec_prj, dec_data = readTif_gdal(dec_path)
dec_data = cv2.resize(dec_data, (raws_y, columns_x))
dec_data = np.array(dec_data, dtype=np.float32)
dec_data[dec_data > 1000] = np.nan
dec_data = dec_data / 1000.0
dec_data_nor = copy.deepcopy(dec_data)

dec_data_nor = np.nanmax(dec_data_nor, axis=2) - np.nanmin(dec_data_nor,axis=2)

deciduousness_ampl = copy.deepcopy(dec_data_nor)

deciduousness_ampl[~forest_mask] = np.nan

ampl_res = deciduousness_ampl.reshape(-1)

data_ampl_y = ampl_res[~final_mask]

def cross_val(sub_model, data_x, data_y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True)
    r2_list = np.array([])
    for train_index, test_index in kf.split(data_x):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        sub_model.fit(x_train, y_train)
        predictions = sub_model.predict(x_test)
        r2 = np.corrcoef(y_test, predictions)[0, 1] ** 2
        r2_list = np.append(r2_list, r2)

    return np.mean(r2_list)

def rf_score(max_depth, learning_rate, subsample, n_estimators):

    xgbr_model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        max_depth=int(max_depth),
        learning_rate=min(learning_rate, 1.0),
        subsample=min(subsample, 1.0),
        n_estimators=int(n_estimators),
        objective='reg:squarederror',
        booster='gbtree',

    )

    r_mean = cross_val(xgbr_model, data_x, data_y, cv=3)

    del xgbr_model

    return r_mean

xgb_bo_for = BayesianOptimization(
    rf_score,
    {'max_depth': (5, 15),
     'learning_rate': (0.01, 1.0),
     'subsample': (0.1, 1.0),
     'n_estimators': (50, 300)
     },
    random_state=100
)

xgb_bo_for.maximize(init_points=0, n_iter=20)

print(xgb_bo_for.max)

best_params = xgb_bo_for.max['params']

best_params = {'learning_rate': 0.18512765385340405, 'max_depth': 12.813679966948698,
               'n_estimators': 147.51467043841544, 'subsample': 0.6025355847978281}

params = {'objective': 'reg:squarederror',
          'booster': 'gbtree',
          'learning_rate': best_params['learning_rate'],
          'max_depth': int(best_params['max_depth']),
          'subsample': best_params['subsample'],
          'n_estimators': int(best_params['n_estimators']),
          'tree_method': 'gpu_hist'}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

predictions_list = np.array([])
test_y_list = np.array([])

shap_values_all = []

for train_index, test_index in kf.split(data_x):
    x_train, x_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]

    model_sub = xgb.XGBRegressor(**params, verbose=0)

    model_sub.fit(x_train, y_train)

    predictions = model_sub.predict(x_test)

    predictions_list = np.append(predictions_list, predictions)
    test_y_list = np.append(test_y_list, y_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = np.corrcoef(y_test.reshape(-1), predictions)
    r2_sk = r2_score(y_test, predictions)

    print('Testing Dataset', " RMSE:", round(np.sqrt(mse), 3), "R2:", round(r2_sk, 3), ' r2', round(r2[0, 1] ** 2, 3))

    predictions = model_sub.predict(x_train)

    r2 = np.corrcoef(y_train.reshape(-1), predictions)
    r2_sk = r2_score(y_train, predictions)

    print('Training Dataset', " RMSE:", round(np.sqrt(mse), 3), "R2:", round(r2_sk, 3), ' r2', round(r2[0, 1] ** 2, 3))

print(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2)
print(round(np.sqrt(mean_squared_error(test_y_list, predictions_list)), 3))

RMSE = round(np.sqrt(mean_squared_error(test_y_list, predictions_list)), 3)

plt.figure(figsize=(6, 5), dpi=150)

plt.hexbin(test_y_list, predictions_list, gridsize=80, mincnt=1, cmap='viridis', alpha=0.5)
cbar = plt.colorbar()
cbar.set_label(r"Counts", size=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$ (month)', fontsize=18)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$ (month)', fontsize=18)
plt.text(4.2, 0.4, 'r$^2$ = ' + str(round(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2, 2)), fontsize=14)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5), dpi=300)

df = pd.DataFrame({
    'Observed Bin': test_y_list.astype(int),
    'Predicted Deltat': predictions_list
})

sc = sns.boxplot(data=df, x='Observed Bin', y='Predicted Deltat',
            palette='viridis', width=0.6, linewidth=1.5,showfliers=False,)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$(month)', fontsize=16)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$(month)', fontsize=16)
plt.text(4.2, 0.4, 'r$^2$ = ' + str(round(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2, 2)), fontsize=14)

plt.xticks(np.arange(0, 7, 1), fontsize=14)

plt.tight_layout()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.figure(figsize=(8, 6), dpi=150)

df = pd.DataFrame({
    'Observed Value': test_y_list.astype(int),
    'Predicted Deltat': predictions_list
})

value_counts = df['Observed Value'].value_counts(normalize=True)
df['Proportion'] = df['Observed Value'].map(value_counts)

cmap = plt.cm.Blues
norm = Normalize(vmin=0, vmax=value_counts.max())

ax = sns.boxplot(
    data=df,
    x='Observed Value',
    y='Predicted Deltat',
    palette=[cmap(norm(prop)) for prop in df.groupby('Observed Value')['Proportion'].first()],
    width=0.6,
    linewidth=1.5,
    showfliers=False
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Density', fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.xticks(range(7), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$ (month)', fontsize=16)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$ (month)', fontsize=16)

corr = np.corrcoef(test_y_list, predictions_list)[0, 1]**2
plt.text(0.05, 0.80, f'r^2 = {corr:.2f}',
         transform=ax.transAxes, fontsize=16,
         )

plt.tight_layout()
plt.show()

save_path = r'outputs/figures/FigS7_cross_validation_0521.png'

model = xgb.XGBRegressor(**params)
model.fit(data_x, data_y)

predictions = model.predict(data_x)

mse = mean_squared_error(data_y, predictions)
r2 = np.corrcoef(data_y.reshape(-1), predictions)
r2_sk = r2_score(data_y, predictions)
print("RMSE:", round(np.sqrt(mse), 3))
print("R-squared Score:", round(r2[0, 1] ** 2, 3))
print("R-squared Score:", r2_sk)

plt.figure(figsize=(6, 5))

plt.hexbin(data_y, predictions, gridsize=100, mincnt=2, cmap='viridis', alpha=0.5)

importance_list = model.feature_importances_
sorted_idx = importance_list.argsort()
plt.figure(figsize=(10, 6))
plt.barh([test_variable_list[x].split('.')[0] for x in sorted_idx], importance_list[sorted_idx])
plt.xlabel('Relative Importance', fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

shap.initjs()
st = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_x)

et = time.time()
print('Caculating Shap value costs: ', round((et - st) / 60, 3), 'min')

abs_shap_vaule = np.abs(shap_values)
per_pixels_shap = np.argmax(abs_shap_vaule, axis=1).reshape(-1)
amazon_features = np.ones_like(deciduousness) * np.nan
amazon_features_res = amazon_features.reshape(-1)

amazon_features_res[~final_mask] = per_pixels_shap
amazon_features = amazon_features_res.reshape(amazon_features.shape)

variable_name_list = [x.split('.')[0] for x in variable_lists]
print(variable_name_list)
amazon_features = amazon_features.astype(np.float32)
amazon_features[~forest_mask] = np.nan

plt.figure()
shap.summary_plot(shap_values, data_x, feature_names=variable_name_list, plot_type="bar", cmap='Greens')

plt.tight_layout()

print(np.nanmean(np.abs(shap_values), axis=0))
important_ind = np.argsort(np.nanmean(np.abs(shap_values), axis=0))[::-1]

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay(g/100g(%))',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand(g/100g(%))', ]

feature_ind_list = important_ind

fig, axes = plt.subplots(4, 2, figsize=(4, 12),dpi=300)
norm1 = mcolors.Normalize(vmin=0, vmax=1500)
im = cm.ScalarMappable(norm=norm1, cmap='coolwarm')

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.180, wspace=0.50, hspace=0.5)

for tmp_ind in np.arange(len(feature_ind_list)):

    feature_ind = feature_ind_list[tmp_ind]
    input_y = shap_values[:, feature_ind]

    input_x = data_x[:, feature_ind]

    if 'moisture' in feature_name[feature_ind]:
        input_x = input_x / 10000.0
    elif 'Clay' in feature_name[feature_ind] or 'Sand' in feature_name[feature_ind]:
        input_x = input_x / 10.0

    threshold = 0.05
    p95 = np.nanpercentile(input_x, 100 - threshold)
    p5 = np.nanpercentile(input_x, threshold)

    input_x_mask = np.logical_and(input_x < p95, input_x > p5)

    fit_x = input_x[input_x_mask]
    fit_y = input_y[input_x_mask]

    fit_degree = 2

    axs = axes.flatten()[tmp_ind]
    axs.hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')

    bins = np.linspace(fit_x.min(), fit_x.max(), 200)

    input_x_bins = np.digitize(fit_x, bins)

    bin_statistics = np.array([np.nanmean(fit_y[input_x_bins == i]) for i in range(1, len(bins) + 1)])

    try:
        model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
    except:
        model_fit = np.poly1d(np.polyfit(bins, bin_statistics, fit_degree))

    axs.plot(bins, model_fit(bins), 'k', linestyle='-', linewidth=1, alpha=0.8)

    axs.set_xlabel(feature_name[feature_ind], fontsize=7)

    if 'Fertility' in feature_name[feature_ind]:
        axs.set_ylabel('SHAP value for Soil fertility\n' +
                       r'Low $\leftarrow$   $\rightarrow$ High', fontsize=7)
        xticks_label = [r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$']
        axs.set_xticks([-1, -0.5, 0, 0.5])
        axs.set_xticklabels(xticks_label, fontsize=7)
    else:
        axs.set_ylabel('SHAP value for {}\n'.format(feature_name[feature_ind].split('(')[0]) +
                       r'Low $\leftarrow$   $\rightarrow$ High',
                       fontsize=7)

    axs.set_yticks([-1, 0.0, 1])
    axs.tick_params(axis='both', which='major', labelsize=6)
    axs.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

cbar_ax = fig.add_axes([0.125, 0.10, 0.8, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Number of samples', fontsize=7)
cbar.ax.tick_params(labelsize=6)

save_path = r'outputs/figures/FigS9_Partial_dependence_plot_4x3_0521.png'

width_x, height_y = 650, 786

feature_name = ['PAR', 'MCWD', 'MAP', 'Herbivory', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand']

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand', ]

color_list = [
    'tab:orange',
    'tab:brown',
    'lightgray',
    'tab:blue',
]

feature_color = [color_list[0], color_list[3], color_list[3], color_list[1], color_list[2], color_list[2],
                 color_list[3], color_list[2]]

feature_importance = np.nanmean(np.abs(shap_values), axis=0)

feature_importance = feature_importance / np.sum(np.array(feature_importance))

sorted_idx = feature_importance.argsort()

type1 = [feature_name.index(x) for x in feature_name if feature_color[feature_name.index(x)] == color_list[0]]
type2 = [feature_name.index(x) for x in feature_name if
         feature_color[feature_name.index(x)] == color_list[3]]
type3 = [feature_name.index(x) for x in feature_name if feature_color[feature_name.index(x)] == color_list[2]]
type4 = [feature_name.index(x) for x in feature_name if
         feature_color[feature_name.index(x)] == color_list[1]]

class_importance = np.zeros(4)

class_importance[0] = np.sum(np.array([feature_importance[x] for x in type3]))
class_importance[1] = np.sum(np.array([feature_importance[x] for x in type2]))
class_importance[2] = np.sum(np.array([feature_importance[x] for x in type1]))
class_importance[3] = np.sum(np.array([feature_importance[x] for x in type4]))

class_importance = np.round(class_importance, 3)

if np.sum(class_importance) != 1:
    index_max = np.argmax(class_importance)
    class_importance[index_max] = 1 - np.sum(class_importance) + class_importance[index_max]

Full_feature_map = copy.deepcopy(amazon_features)

five_types = [type1, type2, type3, type4]

dominant_map = np.zeros((width_x, height_y)) * np.nan

for id in np.arange(len(five_types)):
    ind_list = five_types[id]
    for ind in ind_list:
        ind_x, ind_y = np.where(Full_feature_map == ind)
        dominant_map[ind_x, ind_y] = id

color_list = [
    'tab:orange',
    'tab:brown',
    'lightgray',
    'tab:blue',
]

fig, ax1 = plt.subplots(figsize=(6, 8))

ax1.barh([feature_name[x] for x in sorted_idx], feature_importance[sorted_idx],
         color=[feature_color[x] for x in sorted_idx])

ax1.tick_params(axis='both', which='major', length=3)

ax1.set_xlabel('Feature Importance', fontsize=16)

ax1.tick_params(axis='both', which='major', labelsize=14)

left, bottom, width, height = [0.450, 0.2, 0.45, 0.45]

ax2 = fig.add_axes([left, bottom, width, height])

ax2.pie(class_importance, labels=['Soil', 'Hydroclimate', 'PAR', 'Herbivory'], pctdistance=0.56,
        labeldistance=1.05,
        autopct='%1.1f%%', colors=[color_list[2], color_list[3], color_list[0], color_list[1]],

        textprops={'fontsize': 14, 'color': 'k', 'weight': 'regular'})

plt.tight_layout()
save_path = r'outputs/figures/Feature_importance_herbivory.png'

for i in sorted_idx:
    feature_importance[i] = feature_importance[i] / np.sum(np.array(feature_importance))
    print(feature_name[i], " is ", feature_importance[i])

feature_name_8 = np.array(
    ['SR', 'MCWD', 'MAP', 'Herbivory', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand'])

feature_colors_8 = np.array([
    'tab:orange',
    'tab:blue',
    'tab:blue',
    'tab:brown',
    'lightgray',
    'lightgray',
    'tab:blue',
    'lightgray'
])

imp_entire = np.nanmean(np.abs(shap_values), axis=0)
imp_entire = imp_entire / np.nansum(imp_entire)

num_bins = 3
quantiles = np.linspace(0, 100, num_bins + 1)

bin_edges = np.percentile(data_ampl_y, quantiles)

imp_strata = []
for i in range(num_bins):
    if i == num_bins - 1:
        mask = (data_ampl_y >= bin_edges[i]) & (data_ampl_y <= bin_edges[i + 1])
    else:
        mask = (data_ampl_y >= bin_edges[i]) & (data_ampl_y < bin_edges[i + 1])

    shap_stratum = shap_values[mask, :]

    mean_abs_stratum = np.nanmean(np.abs(shap_stratum), axis=0)
    norm_imp_stratum = mean_abs_stratum / np.nansum(mean_abs_stratum)
    imp_strata.append(norm_imp_stratum)

plot_data = [imp_entire, imp_strata[0], imp_strata[1], imp_strata[2]]
titles = [
    'Entire Basin',
    'Low Amplitude',
    'Medium Amplitude',
    'High Amplitude'
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=200, sharex=True)
plt.subplots_adjust(wspace=0.3)

for i, ax in enumerate(axes):
    current_imp = plot_data[i]

    sorted_idx = current_imp.argsort()

    bars = ax.barh(feature_name_8[sorted_idx], current_imp[sorted_idx],
                   color=feature_colors_8[sorted_idx], edgecolor='white', alpha=0.9)

    ax.set_title(titles[i], fontsize=16, pad=15)
    ax.set_xlabel('Relative Importance', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

plt.tight_layout()

plt.show()

feature_name_bar = ['SR', 'MCWD', 'MAP', 'Herbivory', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand', ]

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand', ]

feature_ind_list = important_ind[:3]

fig, axes = plt.subplots(2, 2, figsize=(6, 5.5),dpi=300)

plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.20, wspace=0.30, hspace=0.4)

norm1 = mcolors.Normalize(vmin=0, vmax=1500)
im = cm.ScalarMappable(norm=norm1, cmap='coolwarm')

for tmp_ind in np.arange(4):
    if tmp_ind == 0:
        ax1 = axes.flatten()[tmp_ind]
        ax1.barh([feature_name_bar[x] for x in sorted_idx], feature_importance[sorted_idx],
                 color=[feature_color[x] for x in sorted_idx])

        ax1.tick_params(axis='both', which='major', length=2)

        ax1.set_xlabel('Feature Importance', fontsize=8)

        ax1.tick_params(axis='both', which='major', labelsize=7)

        pie_left, pie_bottom, pie_width, pie_height = [0.38, 0.03, 0.5, 0.5]

        ax2 = fig.add_axes([
            ax1.get_position().x0 + pie_left * ax1.get_position().width,
            ax1.get_position().y0 + pie_bottom * ax1.get_position().height,
            pie_width * ax1.get_position().width,
            pie_height * ax1.get_position().height
        ])

        wedges, texts, autotexts = ax2.pie(
            class_importance,
            labels=['Soil', 'Hydroclimate', 'Light', 'Herbivory'],
            pctdistance=0.56,
            labeldistance=1.05,
            autopct='%1.1f%%',
            colors=[color_list[2], color_list[3], color_list[0], color_list[1]],
            textprops={'fontsize': 6, 'weight': 'regular'}
        )

    else:
        feature_ind = feature_ind_list[tmp_ind - 1]
        input_y = shap_values[:, feature_ind]

        input_x = data_x[:, feature_ind]

        threshold = 0.05
        p95 = np.nanpercentile(input_x, 100 - threshold)
        p5 = np.nanpercentile(input_x, threshold)

        input_x_mask = np.logical_and(input_x < p95, input_x > p5)

        fit_x = input_x[input_x_mask]
        fit_y = input_y[input_x_mask]

        if tmp_ind == 1:
            ax_id = 2
        elif tmp_ind == 2:
            ax_id = 1
        elif tmp_ind == 3:
            ax_id = 3
        axs = axes.flatten()[ax_id]
        axs.hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')

        fit_degree = 2

        bins = np.linspace(fit_x.min(), fit_x.max(), 200)

        input_x_bins = np.digitize(fit_x, bins)

        bin_statistics = np.array([np.nanmean(fit_y[input_x_bins == i]) for i in range(1, len(bins) + 1)])

        try:
            model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
        except:
            model_fit = np.poly1d(
                np.polyfit(bins, bin_statistics, fit_degree))

        axs.plot(bins, model_fit(bins), 'k', linestyle='-', linewidth=1.0, alpha=0.8)

        axs.set_xlabel(feature_name[feature_ind], fontsize=7)

        if 'Fertility' in feature_name[feature_ind]:
            axs.set_ylabel('SHAP value for Soil fertility\n' +
                           r'Low asynchrony $\leftarrow$      $\rightarrow$ High asynchrony',
                           fontsize=8)
            xticks_label = [r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$']
            axs.set_xticks([-1, -0.5, 0, 0.5])
            axs.set_xticklabels(xticks_label, fontsize=7)
        else:
            axs.set_ylabel('SHAP value for {}\n'.format(feature_name[feature_ind].split('(')[0]) +
                           r'Low asynchrony $\leftarrow$      $\rightarrow$ High asynchrony', fontsize=8)

        axs.set_yticks([-1, 0.0, 1])

        axs.tick_params(axis='both', which='major', labelsize=7)

        axs.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

cax = fig.add_axes([0.13, 0.075, 0.8, 0.03])

cbar = fig.colorbar(im, cax=cax, shrink=1.2, fraction=0.08, aspect=30, pad=0.05,
                    orientation='horizontal', )
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Number of samples', fontsize=8)

save_path = r'outputs/figures/Fig4_driver_Vegetation_removal_0818.png'

width_x, height_y = 650, 786

abs_shap_vaule = np.abs(shap_values)

type1_shap = abs_shap_vaule[:, type1].reshape(-1)
type2_shap = np.nansum(abs_shap_vaule[:, type2], axis=1)
type3_shap = np.nansum(abs_shap_vaule[:, type3], axis=1)
type4_shap = np.nansum(abs_shap_vaule[:, type4], axis=1)

drivers_top3_map = np.zeros((width_x, height_y, 3)) * np.nan

drivers_top3_map_res = drivers_top3_map.reshape(-1, 3)

drivers_top3_map_res[~final_mask, 0] = type1_shap
drivers_top3_map_res[~final_mask, 1] = type2_shap
drivers_top3_map_res[~final_mask, 2] = type3_shap

drivers_top3_map = drivers_top3_map_res.reshape(drivers_top3_map.shape)

drivers_top3_map_sum = np.nansum(drivers_top3_map, axis=2)
drivers_top3_map_sum[drivers_top3_map_sum == 0] = np.nan
drivers_top3_map_nor = drivers_top3_map / drivers_top3_map_sum[:, :, np.newaxis]

save_path = r'data/drivers/Asynchrony_shap_map_3type_drivers_0818.tif'
save_tif(drivers_top3_map_nor, save_path, _geo, _prj, 3)
