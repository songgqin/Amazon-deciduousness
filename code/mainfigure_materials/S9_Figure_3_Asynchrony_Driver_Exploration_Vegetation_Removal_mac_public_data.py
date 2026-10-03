from bayes_opt import BayesianOptimization
import seaborn as sns
from scipy import stats
from S1_Data_Preprocessing_amazon import readTif_gdal  # , save_tif
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
from scipy import ndimage

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
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 150


# In[]
def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    datatype = gdal.GDT_Float32
    # datatype = gdal.GDT_UInt16
    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype)

    # if (outputData != None):
    outputData.SetGeoTransform(Geo_)  # 写入仿射变换参数
    outputData.SetProjection(Projection_)  # 写入投影
    # print('done')

    # Write in the DataValue
    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)
        # outputData.GetRasterBand(1).SetNoDataValue(9999)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(1).SetNoDataValue(np.nan)
            # outputData.GetRasterBand(i + 1).SetNoDataValue(9999)
    del outputData


def remove_outliers(datax, datay):
    # Create a DataFrame from datax and datay
    df = pd.DataFrame({'datax': datax, 'datay': datay})

    # Group by datax and calculate mean and standard deviation for each group
    group_stats = df.groupby('datax')['datay'].agg(['mean', 'std'])

    # Calculate lower and upper bounds for outlier removal
    group_stats['lower_bound'] = group_stats['mean'] - 3 * group_stats['std']
    group_stats['upper_bound'] = group_stats['mean'] + 3 * group_stats['std']

    # Merge the statistics back to the original DataFrame
    df = df.merge(group_stats, left_on='datax', right_index=True)

    # Filter out outliers
    mask = (df['datay'] >= df['lower_bound']) & (df['datay'] <= df['upper_bound'])
    df_filtered = df[mask]

    return df_filtered['datax'].values, df_filtered['datay'].values


# In[]
raws_y, columns_x = 786, 650

cls_modis_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\Forest_Mask\MCD12Q1_Amazon.tif'
# cls_modis_path = r'/Volumes/Seagate Basic/PhD_Works/Work4_Amazon_Pattern_Detection_ECAE/Main_Figures/Fig1/Data/Forest_Mask/MCD12Q1_Amazon.tif'

# cls_modis_path = r'X:\Song_Amazon_Mapping\classification_map.tif'
_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)

cls_md_forest[cls_md_forest != 2] = -1
cls_md_forest[cls_md == 0] = np.nan

cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = copy.deepcopy(cls_modis_coarse == 2)

# print(len(forest_mask[~np.isnan(forest_mask)]))
# print(len(np.where(forest_mask == 1)[0]))

# In[] Variable reading
# variables_dir = r'/Users/song/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/PhD_Projects/Project4_Mapping_Amazon_Basin_Using_GEE/Manuscript/Data_Availablity/Data/Environmental_Variables'
# variables_dir = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables'
variables_dir = r'D:\OneDrive - The University Of Hong Kong\PhD_Projects\Project4_Mapping_Amazon_Basin_Using_GEE\Manuscript\Data_Availablity\Data\Environmental_Variables'

variable_lists = [x for x in os.listdir(variables_dir) if x.endswith('.tif')]

variable_lists.sort()

test_variable_list = variable_lists
variables_all = np.zeros((650, 786, len(test_variable_list)))  # 650, 786


for variable_name in test_variable_list:
    _geo, _prj, variable = readTif_gdal(os.path.join(variables_dir, variable_name))
    variable = variable.astype(np.float32)
    # variable[variable < 0] = np.nan
    if 'precipitation' in variable_name:
        variable_mean = np.nansum(variable, axis=2) * 1000.0
        # variable_mean[variable_mean>4SR000] = np.nan
    elif 'par' in variable_name:
        variable_mean = np.nanmean(variable, axis=2)  # * 0.1
        variable_mean[variable_mean < 100] = np.nan  # remove the abnormal value
    elif 'SR' in variable_name:
        variable_mean = np.nanmean(variable, axis=2)  # * 0.1
        variable_mean[variable_mean < 100] = np.nan  # remove the abnormal value
    elif 'temperature' in variable_name:
        variable_mean = np.nanmean(variable, axis=2) - 273.15
    elif 'vpd' in variable_name:
        # variable_mean = np.nanmean(variable, axis=2) * 0.1
        variable_mean = variable  # [:, :, 0]
        # variable_mean = cv2.resize(variable_mean, (786, 650), interpolation=cv2.INTER_LINEAR)  # 786, 650
        # variable_mean[~forest_mask] = np.nan

        # variable_mean[variable_mean < 0] = np.nan

        # plt.imshow(variable_mean)

    # elif 'moisture' in variable_name:
    #     variable_mean = np.nanmean(variable, axis=2) * 0.1
    elif 'wtd' in variable_name:
        variable_mean = -variable
        # variable_mean[variable_mean > 1000] = np.nan
    elif 'sand' in variable_name or 'clay' in variable_name:
        variable_mean = np.nanmean(variable, axis=2)

    elif 'MCWD' in variable_name:
        variable_mean = copy.deepcopy(variable)
        variable_mean[variable_mean < -800] = np.nan
    # elif 'fertility' in variable_name:
    #     variable_mean = 10**variable
    else:
        variable_mean = variable

    # if len(variable.shape) > 2:
    #     if variable.shape[0] > 20:
    #         variable_mean = np.nansum(variable, axis=2)
    #     else:
    #         variable_mean = np.nansum(variable, axis=0)
    # else:
    #     variable_mean = variable

    if 'id' in variable_name:
        variable_mean[variable_mean==0] = np.nan
        variable_mean = cv2.resize(variable_mean, (786, 650), interpolation=cv2.INTER_NEAREST)
    else:
        variable_mean = cv2.resize(variable_mean, (786, 650), interpolation=cv2.INTER_LINEAR)  # 786, 650
    # plt.figure()
    # plt.imshow(variable_mean)

    variable_mean[~forest_mask] = np.nan

    # save_path = os.path.join(save_dir, variable_name.replace('.tif', '_visualized.tif'))
    # save_tif(variable_mean, save_path, _geo, _prj, 1)

    variables_all[:, :, test_variable_list.index(variable_name)] = variable_mean

variable_reshape = variables_all.reshape(-1, variables_all.shape[2])

# variable_reshape = variable_reshape[forest_mask.reshape(-1)]

# plt.imshow(annual_variable)
# plt.colorbar()
# plt.show()
# In[] calculate the variance inflation factor (VIF) values for each variable

# add a constant term to the data
variable_name_list = [x.split('.')[0] for x in test_variable_list]

variable_reshape_vif = copy.deepcopy(variable_reshape.astype(np.float64))

# remove the row with nan value and inf value
inf_mask = np.any(np.isinf(variable_reshape_vif), axis=1)
nan_mask = np.any(np.isnan(variable_reshape_vif), axis=1)
final_mask = np.logical_or(inf_mask, nan_mask)

variable_reshape_vif = variable_reshape_vif[~final_mask]

pd_data = pd.DataFrame(variable_reshape_vif, columns=variable_name_list)

# calculate VIF for each explanatory variable
vif_data = add_constant(pd_data)
vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])],
                index=vif_data.columns)

# In[] Read the deciduousness


# deciduousness_path = r'/Volumes/Seagate Basic/PhD_Works/Work4_Amazon_Pattern_Detection_ECAE/Main_Figures/Fig2/Python-Based Figures/Tif_Data/time_lag_map_0521.tif'  #
# deciduousness_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\Cor_Dec_EVI_0521.tif'  #
deciduousness_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map_0521.tif'  #


_geo, _prj, deciduousness = readTif_gdal(deciduousness_path)

# annual_deciduousness = ndimage.median_filter(
#     deciduousness,
#     size=3,
#     mode='constant',
#     cval=np.nan  # SciPy 能正确处理 NaN
# )

# annual_deciduousness[~valid_mask] = np.nan
# annual_deciduousness = cv2.resize(deciduousness, (annual_precipitation.shape[1], annual_precipitation.shape[0]))
# annual_deciduousness = copy.deepcopy(deciduousness / 1000.0)
annual_deciduousness = copy.deepcopy(deciduousness)
# annual_deciduousness = cv2.resize(annual_deciduousness, (786, 650))  # 786, 650
# 4. 恢复原始 NaN 掩码（只保留森林区域数据）
# annual_deciduousness[~forest_mask] = np.nan
annual_deciduousness = cv2.medianBlur(annual_deciduousness, 3)
# annual_deciduousness[annual_deciduousness==0] = np.nan

# normalize the deciduousness
# annual_deciduousness = (annual_deciduousness - np.nanmin(annual_deciduousness)) / ( np.nanmax(annual_deciduousness) - np.nanmin(annual_deciduousness))

# plt.imshow(annual_deciduousness)

# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\deciduousness_ampl_0919.tif'
# save_tif(annual_deciduousness, save_path, _geo, _prj, 1)

# kernel = morphology.disk(3)
# annual_deciduousness = morphology.opening(annual_deciduousness, kernel)

# mask_nan = np.isnan(annual_precipitation)
# annual_deciduousness[mask_nan] = np.nan

# annual_deciduousness[annual_deciduousness > 1] = 1
# annual_deciduousness[annual_deciduousness < 0] = 0

# In[] construct the dataset for regression model
# min_max_scaler = MinMaxScaler()

deciduousness_res = annual_deciduousness.reshape(-1)
# raw data
inf_mask = np.any(np.isinf(variable_reshape), axis=1)
nan_mask = np.logical_or(np.isnan(deciduousness_res), np.any(np.isnan(variable_reshape), axis=1))

final_mask = np.logical_or(inf_mask, nan_mask)
# final_mask = copy.deepcopy(inf_mask )
# data_y = deciduousness_res[~np.isnan(deciduousness_res)]
# data_x = variable_reshape[~np.isnan(deciduousness_res)]
# data_y_raw = deciduousness_res[~nan_mask]
# data_x_raw = variable_reshape[~nan_mask]

data_y = deciduousness_res[~final_mask]

# print(np.sum(data_y > 0.1) / len(data_y))
# data_x_raw = variable_reshape[~final_mask]
# scaler = StandardScaler()
# data_x = scaler.fit_transform(data_x_raw)
data_x = variable_reshape[~final_mask]


# inf_mask = np.any(np.isinf(data_x_raw),axis=1)

# In[] linear regression model
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_x[:,1:])  # (x - mean) / std

# Create a DataFrame for the independent variables
df_x = pd.DataFrame(X_scaled, columns=variable_name_list[1:])
# Create a DataFrame for the dependent variable
df_y = pd.DataFrame(data_y, columns=['deciduousness'])
# Combine the independent and dependent variables into a single DataFrame
df = pd.concat([df_y, df_x], axis=1)

# Fit the linear  model
model = sm.OLS(df_y,df_x).fit()
# model = smf.ols('deciduousness ~ ' + ' + '.join(variable_name_list), data=df).fit()
# model = LinearRegression()
# model.fit(df_x, df_y)
# Get the coefficients and intercept
# coefficients = model.coef_[0]  # Get the coefficients of the model
# intercept = model.intercept_[0]  # Get the intercept of the model
# Create a summary DataFrame
# Print the summary of the model
print(model.summary())

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay(g/100g(%))',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand(g/100g(%))', ]
# visualize the feature importance
feature_importance = model.params[1:-1].tolist()  # Exclude the intercept
rank_ind = np.argsort(np.abs(feature_importance)) # Sort by absolute value
plt.figure(figsize=(10, 6))
plt.barh([feature_name[x] for x in rank_ind], feature_importance[rank_ind])
# plt.barh([feature_name[x] for x in rank_ind], feature_importance)
plt.xlabel('Coefficient Value', fontsize=18)
plt.ylabel('Variables', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# In[] linear mixed effect model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_x[:,1:])  # (x - mean) / std


# Create a DataFrame for the independent variables
df_x = pd.DataFrame(X_scaled, columns=variable_name_list[1:])
# Add the group variable (e.g., grid_id) to the DataFrame
df_x['group'] = data_x[:, 0]  # Assuming the second column is the group variable
# Create a DataFrame for the dependent variable
df_y = pd.DataFrame(data_y, columns=['deciduousness'])
# Combine the independent and dependent variables into a single DataFrame
df = pd.concat([df_y, df_x], axis=1)

# Fit the linear mixed effects model
model = smf.mixedlm('deciduousness ~ ' + ' + '.join(variable_name_list[1:]), data=df, groups=df['group']).fit()

print("\n 混合线性模型结果:")
print(model.summary())

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay(g/100g(%))',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand(g/100g(%))', ]
# visualize the feature importance
feature_importance = model.params[1:-1]  # Exclude the intercept
rank_ind = np.argsort(np.abs(feature_importance)) # Sort by absolute value
plt.figure(figsize=(10, 6))
plt.barh([feature_name[x] for x in rank_ind], feature_importance[rank_ind])
plt.xlabel('Coefficient Value', fontsize=18)
plt.ylabel('Variables', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()


# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\df_input.csv'
# df.to_csv(save_path, index=False)

# In[] optimze the hyperparameter
def cross_val(sub_model, data_x, data_y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True)  # five-fold evaluation
    r2_list = np.array([])
    for train_index, test_index in kf.split(data_x):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        sub_model.fit(x_train, y_train)
        predictions = sub_model.predict(x_test)
        r2 = np.corrcoef(y_test, predictions)[0, 1] ** 2
        r2_list = np.append(r2_list, r2)

    return np.mean(r2_list)


def rf_score(max_depth, learning_rate, subsample, n_estimators):  #

    xgbr_model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        max_depth=int(max_depth),
        learning_rate=min(learning_rate, 1.0),
        subsample=min(subsample, 1.0),
        n_estimators=int(n_estimators),
        objective='reg:squarederror',
        booster='gbtree',
        # gpu_id=0
    )
    # dtrain = xgb.DMatrix(Xfor_train, label=Yfor_train)
    # xgbr_model.fit(data_x, data_y)
    # predictions = xgbr_model.predict(data_x)
    # r2 = np.corrcoef(data_y, predictions)[0,1]**2
    r_mean = cross_val(xgbr_model, data_x, data_y, cv=3)

    del xgbr_model

    return r_mean  # cross_val_score(xgbr_model, data_x, data_y, cv=3, scoring="r2").mean() #


xgb_bo_for = BayesianOptimization(
    rf_score,
    {'max_depth': (5, 15),
     'learning_rate': (0.01, 1.0),
     'subsample': (0.1, 1.0),
     'n_estimators': (50, 300)
     },
    random_state=100
)

# xgb_bo_for.set_gp_params(alpha=1e-3)
xgb_bo_for.maximize(init_points=0, n_iter=20)

print(xgb_bo_for.max)

best_params = xgb_bo_for.max['params']

# In[] Relationship between deciduousness and variable
#
# for index, name in enumerate(variable_lists):
#     # plt.figure()
#     # plt.scatter(data_x[:, index], data_y)
#     # plt.title(name, fontsize=16)
#     print(name, round(np.corrcoef(data_x[:, index], data_y)[0, 1] , 3))

# In[] hyperparameter tuning
# param_grid = {
#     'max_depth': [6,7,8,9],
#     'learning_rate': [0.3, 0.03, 0.003],
#     'subsample': [0.5, 0.7, 1],
#     'n_estimators': [500, 800, 1000],
# }
# xgb_model = xgb.XGBRegressor(tree_method='gpu_hist')
# grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2')
#
# grid_search.fit(data_x, data_y)


# In[] build up the regression model and extract the relative importance of each climate variable

# best_params = {'learning_rate': 0.09905833790611081, 'max_depth': 12.889558578255013, 'n_estimators': 138.3123171733379, 'subsample': 0.5385016826033913}
# best_params = {'learning_rate': 0.017159208068903774, 'max_depth': 14.879018969360946, 'n_estimators': 289.4908864344218, 'subsample':0.9822443606973412}
# best_params = {'learning_rate': 0.07680668234371166, 'max_depth': 11.709051763327043,
#                'n_estimators': 249.87104087800324, 'subsample': 0.6220195461736212}

best_params = {'learning_rate': 0.18512765385340405, 'max_depth': 12.813679966948698,
               'n_estimators': 147.51467043841544, 'subsample': 0.6025355847978281}
# best_params =  {'learning_rate': 0.01, 'max_depth': 15.0, 'n_estimators': 157.82362259353076, 'subsample': 1.0}

params = {'objective': 'reg:squarederror',
          'booster': 'gbtree',
          'learning_rate': best_params['learning_rate'],
          'max_depth': int(best_params['max_depth']),
          'subsample': best_params['subsample'],
          'n_estimators': int(best_params['n_estimators'])}

# params = {'objective': 'reg:squarederror',
#           'booster': 'gbtree',
#           'learning_rate': 0.15652707975253924,
#           'max_depth': 13,
#           'subsample': 0.8082477241194193,
#           'n_estimators': 138,
#           'tree_method': 'gpu_hist'}

# In[]
# five fold evaluation
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # five-fold evaluation
# X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1)

predictions_list = np.array([])
test_y_list = np.array([])

shap_values_all = []

for train_index, test_index in kf.split(data_x):
    x_train, x_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]

    # plt.figure()
    # plt.hist(y_test)

    model_sub = xgb.XGBRegressor(**params, verbose=0)  # best_params params
    # model_sub = RandomForestRegressor(n_estimators=100)
    model_sub.fit(x_train, y_train)

    # calculate the shap value of each sample
    # shap.initjs()
    # explainer = shap.TreeExplainer(model_sub)  # model
    # shap_values = explainer.shap_values(data_x)
    #
    # shap_values_all.append(shap_values)

    # -----------Testing-------------------
    predictions = model_sub.predict(x_test)

    predictions_list = np.append(predictions_list, predictions)
    test_y_list = np.append(test_y_list, y_test)

    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_test, predictions)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title('Testing')

    mse = mean_squared_error(y_test, predictions)
    r2 = np.corrcoef(y_test.reshape(-1), predictions)
    r2_sk = r2_score(y_test, predictions)

    # predictions = model_sub.predict(x_train)

    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_test, predictions)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title('Testing')

    print('Testing Dataset', " RMSE:", round(np.sqrt(mse), 3), "R2:", round(r2_sk, 3), ' r2', round(r2[0, 1] ** 2, 3))

    # -----------Training-------------------
    predictions = model_sub.predict(x_train)

    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_train, predictions)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title('Training')

    r2 = np.corrcoef(y_train.reshape(-1), predictions)
    r2_sk = r2_score(y_train, predictions)

    print('Training Dataset', " RMSE:", round(np.sqrt(mse), 3), "R2:", round(r2_sk, 3), ' r2', round(r2[0, 1] ** 2, 3))

    # del model_sub

print(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2)
print(round(np.sqrt(mean_squared_error(test_y_list, predictions_list)), 3))
# r2_score(predictions_list, test_y_list)
RMSE = round(np.sqrt(mean_squared_error(test_y_list, predictions_list)), 3)

# In[]
plt.figure(figsize=(6, 5))
# plt.scatter(test_y_list, predictions_list)
plt.hexbin(test_y_list, predictions_list, gridsize=80, mincnt=1, cmap='viridis', alpha=0.5)
cbar = plt.colorbar()
cbar.set_label(r"Counts", size=16)
# plt.title('10-fold Cross Validation', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$ (month)', fontsize=18)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$ (month)', fontsize=18)
plt.text(4.2, 0.4, 'r$^2$ = ' + str(round(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2, 2)), fontsize=14)
# plt.text(4.2, 0.15, 'RMSE = ' + str(RMSE), fontsize=14)
# plt.plot([0, 1], [0, 1], 'k--')
plt.tight_layout()
# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Supplementary_Figures\FigS7_cross_validation_0521.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')


# In[] box plot to show the distribution of the predicted deciduousness
plt.figure(figsize=(6, 5))

df = pd.DataFrame({
    'Observed Bin': test_y_list.astype(int),#pd.cut(test_y_list, bins=np.arange(0, 7, 1)),
    'Predicted Δt': predictions_list
})

sc = sns.boxplot(data=df, x='Observed Bin', y='Predicted Δt',
            palette='viridis', width=0.6, linewidth=1.5,showfliers=False,)  # hide the outliers)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$(month)', fontsize=16)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$(month)', fontsize=16)
plt.text(4.2, 0.4, 'r$^2$ = ' + str(round(np.corrcoef(test_y_list, predictions_list)[0, 1] ** 2, 2)), fontsize=14)

# ticks int
plt.xticks(np.arange(0, 7, 1), fontsize=14)

plt.tight_layout()

# In[]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.figure(figsize=(8, 6))

# 创建DataFrame
df = pd.DataFrame({
    'Observed Value': test_y_list.astype(int),  # 确保是0-6的整数
    'Predicted Δt': predictions_list
})

# 计算每个观测值的占比
value_counts = df['Observed Value'].value_counts(normalize=True)
df['Proportion'] = df['Observed Value'].map(value_counts)

# 创建颜色映射 (从浅色到深色)
cmap = plt.cm.Blues  # 可以使用其他颜色映射如'Reds','Greens'等
norm = Normalize(vmin=0, vmax=value_counts.max())

# 绘制箱线图
ax = sns.boxplot(
    data=df,
    x='Observed Value',
    y='Predicted Δt',
    palette=[cmap(norm(prop)) for prop in df.groupby('Observed Value')['Proportion'].first()],
    width=0.6,
    linewidth=1.5,
    showfliers=False
)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Density', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# # 在柱子上标注百分比
# for i, val in enumerate(range(7)):
#     if val in value_counts.index:
#         prop = value_counts[val]
#         ax.text(i, ax.get_ylim()[0]-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0]),
#                 f'{prop:.1%}',
#                 ha='center', va='top', fontsize=10)

# 设置图形属性
plt.xticks(range(7), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'Observed $\triangle$$\mathit{t}$ (month)', fontsize=16)
plt.ylabel(r'Predicted $\triangle$$\mathit{t}$ (month)', fontsize=16)

# 添加r²值
corr = np.corrcoef(test_y_list, predictions_list)[0, 1]**2
plt.text(0.05, 0.80, f'r² = {corr:.2f}',
         transform=ax.transAxes, fontsize=16,)

plt.tight_layout()
plt.show()

save_path = r'/Users/song/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/PhD_Projects/Project4_Mapping_Amazon_Basin_Using_GEE/Manuscript/Python_output_figures/FigS7_cross_validation_0521.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] All data combined

# params = {'objective': 'reg:squarederror',
#           'booster': 'gblinear',
#           'min_child_weight': 5,
#           'n_estimators': 1000,
#           'eval_metric': 'rmse',
#           'learning_rate': 0.3,
#           'max_depth': 10,
#           'subsample': 0.8,
#           'tree_method': 'gpu_hist'}

# params = {'objective': 'reg:squarederror',
#           'n_estimators': 200,
#           'booster': 'gbtree',
#           'colsample_bytree': 0.8,
#           # 'min_child_weight': 5,
#
#           # 'eval_metric': 'rmse',
#           # 'learning_rate': 0.3,
#           'max_depth': 8,
#           'subsample': 0.8,
#           # 'gmma': 0.1,
#           # 'min_child_weight': 3,
#           #   'lambda': 2,
#           'tree_method': 'gpu_hist'}

# params = {'objective': 'reg:squarederror',
#           'n_estimators': 150,
#           'booster': 'gbtree',
#           # 'colsample_bytree': 0.8,
#           # 'min_child_weight': 5,
#
#           # 'eval_metric': 'rmse',
#           # 'learning_rate': 0.3,
#           'max_depth': 10,
#           # 'subsample': 0.8,
#           # 'gmma': 0.1,
#           # 'min_child_weight': 3,
#           #   'lambda': 2,
#           'tree_method': 'gpu_hist'}
model = xgb.XGBRegressor(**params)
model.fit(data_x, data_y)
# Making predictions on the test set
predictions = model.predict(data_x)
# Calculate the mean squared error and R-squared score
mse = mean_squared_error(data_y, predictions)
r2 = np.corrcoef(data_y.reshape(-1), predictions)
r2_sk = r2_score(data_y, predictions)
print("RMSE:", round(np.sqrt(mse), 3))
print("R-squared Score:", round(r2[0, 1] ** 2, 3))
print("R-squared Score:", r2_sk)

plt.figure(figsize=(6, 5))
# plt.scatter(data_y, predictions)
plt.hexbin(data_y, predictions, gridsize=100, mincnt=2, cmap='viridis', alpha=0.5)

# plt.plot([0, 1], [0, 1], 'k--')

# del model
# In[] plot the relative importance of each climate variable
importance_list = model.feature_importances_
sorted_idx = importance_list.argsort()
plt.figure(figsize=(10, 6))
plt.barh([test_variable_list[x].split('.')[0] for x in sorted_idx], importance_list[sorted_idx])
plt.xlabel('Relative Importance', fontsize=18)
# plt.ylabel('Variables', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# In[] Shap value for each sample -- relative importance for each sample -- spatial
shap.initjs()
st = time.time()
explainer = shap.TreeExplainer(model)  # model
shap_values = explainer.shap_values(data_x)
# shap_explain = explainer(data_x)
et = time.time()
print('Caculating Shap value costs: ', round((et - st) / 60, 3), 'min')

# In[]
abs_shap_vaule = np.abs(shap_values)
per_pixels_shap = np.argmax(abs_shap_vaule, axis=1).reshape(-1)

amazon_features = np.ones_like(deciduousness) * np.nan
amazon_features_res = amazon_features.reshape(-1)
#
# amazon_features_res[~np.isnan(deciduousness_res)] = per_pixels_shap
amazon_features_res[~final_mask] = per_pixels_shap
amazon_features = amazon_features_res.reshape(amazon_features.shape)

# save_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\shap_feature.tif'
# save_tif(amazon_features, save_path, _geo, _prj, 1)
# print(variable_lists)

# shap.force_plot(explainer.expected_value, shap_values[1], data_x[1,:])
variable_name_list = [x.split('.')[0] for x in variable_lists]
print(variable_name_list)

amazon_features = amazon_features.astype(np.float32)
amazon_features[~forest_mask] = np.nan

# save_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\Feature_importance\Dominant_asynchrony_shap_feature.tif'
# save_tif(amazon_features, save_path, _geo, _prj, 1)

# shap_values_all = np.nanmean( np.array(shap_values_all), axis=0)

# In[] Plot the summary plot of the shap value
# shap_values = copy.deepcopy(shap_values_all)
# variable_name_list = ['Canopy Height', 'Water Table Depth','PAR', 'Precipitation',  'Soil Clay', 'Soil Fertility', 'Soil Sand'] # 'VPD',
plt.figure()
shap.summary_plot(shap_values, data_x, feature_names=variable_name_list, plot_type="bar", cmap='Greens')  #
# plt.xlabel('SHAP Value (Feature Importance)', fontsize=16)
plt.tight_layout()
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
print(np.nanmean(np.abs(shap_values), axis=0))
important_ind = np.argsort(np.nanmean(np.abs(shap_values), axis=0))[::-1]

# shap.summary_plot(shap_values,data_x, feature_names=variable_name_list)


# shape value rerank map -- for the top 3 feature and use the proportion of the shap value to show the relative importance
abs_shap_vaule_top3 = np.abs(shap_values[:, important_ind[:3]]) # MWCD; SOIL; LIGHT
shap_value_proportion = abs_shap_vaule_top3 / np.nansum(abs_shap_vaule_top3, axis=1, keepdims=True)

amazon_features_3top = np.ones((deciduousness.shape[0], deciduousness.shape[1], 3)) * np.nan

amazon_features_3top_res = amazon_features_3top.reshape(-1, 3)

amazon_features_3top_res[~final_mask, :] = shap_value_proportion

amazon_features_3top = amazon_features_3top_res.reshape(amazon_features_3top.shape)







# In[] FigS for each feature 4*3 -- Supplementary figure-- partial dependence plot for each feature

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay(g/100g(%))',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand(g/100g(%))', ]  # 'DSL (month)',
# "10-based logarithms"
# feature_ind_list = important_ind[:3]  # [:3]
feature_ind_list = important_ind

# fig, axs = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
fig, axes = plt.subplots(4, 2, figsize=(4, 12))  # , layout="constrained"
norm1 = mcolors.Normalize(vmin=0, vmax=1500)
im = cm.ScalarMappable(norm=norm1, cmap='coolwarm')

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.180, wspace=0.50, hspace=0.5)

# tmp_ind = 3
for tmp_ind in np.arange(len(feature_ind_list)):

    feature_ind = feature_ind_list[tmp_ind]
    input_y = shap_values[:, feature_ind]
    # input_y = copy.deepcopy(data_y)
    input_x = data_x[:, feature_ind]

    if 'moisture' in feature_name[feature_ind]:
        input_x = input_x / 10000.0
    elif 'Clay' in feature_name[feature_ind] or 'Sand' in feature_name[feature_ind]:
        input_x = input_x / 10.0

    # select the data within the range of 5% to 95% percentile
    threshold = 0.05
    p95 = np.nanpercentile(input_x, 100 - threshold)
    p5 = np.nanpercentile(input_x, threshold)

    input_x_mask = np.logical_and(input_x < p95, input_x > p5)

    # if 'PAR' in feature_name[feature_ind]:
    #     input_x_mask = input_x > 160

    fit_x = input_x[input_x_mask]
    fit_y = input_y[input_x_mask]

    fit_degree = 2
    # if 'VPD' in feature_name[feature_ind]:
    #     fit_degree = 3
    # elif 'MAP' in feature_name[feature_ind]:
    #     fit_degree = 2
    #     fit_x = fit_x * 1000
    # # elif 'PAR' in feature_name[feature_ind]:
    # #     fit_degree = 3
    # # elif 'Herbivory' in feature_name[feature_ind]:
    # #     fit_x[fit_x > 10] = np.nan
    # else:
    #     fit_degree = 2
    # axs[i, j].hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')
    axs = axes.flatten()[tmp_ind]
    axs.hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')

    # model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
    bins = np.linspace(fit_x.min(), fit_x.max(), 200)

    input_x_bins = np.digitize(fit_x, bins)
    #
    bin_statistics = np.array([np.nanmean(fit_y[input_x_bins == i]) for i in range(1, len(bins) + 1)])
    # bin_statistics_std_up = [np.nanpercentile(input_y[input_x_bins == i], 95) for i in range(1, len(bins) + 1)]
    # bin_statistics_std_down = [np.nanpercentile(input_y[input_x_bins == i], 5) for i in range(1, len(bins) + 1)]
    try:
        model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
    except:
        model_fit = np.poly1d(np.polyfit(bins, bin_statistics, fit_degree))
    # if 'PAR' in feature_name[feature_ind]:
    #     model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
    # else:
    #     model_fit = np.poly1d(np.polyfit(bins, bin_statistics, fit_degree))

    axs.plot(bins, model_fit(bins), 'k', linestyle='-', linewidth=1, alpha=0.8)

    # axs.plot(bins, model_fit(bins), 'k', linestyle='-', linewidth=1.5, alpha=0.8)

    # sns.regplot(x=fit_x, y=fit_y,  ci=95, line_kws={'linestyle': '-', 'color': 'k'},
    #             ax=axs[i, j],scatter=False, order=fit_degree)

    # axs[i, j].plot(bins, bin_statistics, 'o', )  # color=feature_color[feature_ind])  # Plot mean deciduousness amplitude against bin edges

    # plt.fill_between(bins, bin_statistics_std_up, bin_statistics_std_down, alpha=0.2, color=feature_color[feature_ind])
    # plt.plot(bins, bin_statistics, 'o', )  # color=feature_color[feature_ind])  # Plot mean deciduousness amplitude against bin edges
    # sns.regplot(x=bins, y=bin_statistics, order=1, ci=90, scatter_kws={'s': 40, 'alpha': 0.6},
    #             line_kws={'linestyle': '--', 'color': 'k'})

    # axs[i, j].set_title(feature_name[feature_ind], fontsize=18)
    axs.set_xlabel(feature_name[feature_ind], fontsize=7)

    # if 'Fertility' in feature_name[feature_ind]:
    #     axs.set_ylabel('SHAP value for Soil fertility', fontsize=7)
    #     xticks_label = ['10$^{-1}$', '10$^{-0.5}$', '10$^0$', '10$^{0.5}$']
    #     axs.set_xticks([-1, -0.5, 0, 0.5])
    #     axs.set_xticklabels(xticks_label, fontsize=7)
    #
    # else:
    #     axs.set_ylabel('SHAP value for {}'.format(feature_name[feature_ind].split('(')[0]), fontsize=7)
    # Special handling for Fertility feature
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

        # r'Low asynchrony $\leftarrow$      $\rightarrow$ High asynchrony',

    # if feature_name[feature_ind] == 'PAR':
    #     axs[i, j].set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    #     axs[i, j].set_yticklabels([-0.2, -0.1, 0.0, 0.1, 0.2])
    # elif feature_name[feature_ind] == 'Precipitation':
    #     axs[i, j].set_yticks([ -0.1, 0.0, 0.1, 0.2, 0.3])
    #     axs[i, j].set_yticklabels([ -0.1, 0.0, 0.1, 0.2, 0.3])
    # else:
    #     axs[i, j].set_yticks([ -0.1, 0.0, 0.1, 0.2])
    #     axs[i, j].set_yticklabels([ -0.1, 0.0, 0.1, 0.2])
    # if feature_name[feature_ind] == 'VPD':
    #     axs[i, j].set_yticks([ -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    #     axs[i, j].set_yticklabels([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    #
    # else:
    #     axs[i, j].set_yticks([  -1.0, 0.0, 1.0, 2.0])
    #     axs[i, j].set_yticklabels([ -1.0, 0.0, 1.0, 2.0])
    axs.set_yticks([-1, 0.0, 1])
    axs.tick_params(axis='both', which='major', labelsize=6)
    axs.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # if 'XXX' in feature_name[feature_ind]:
    #     axs.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # elif 'MAP' in feature_name[feature_ind]:
    #     axs.set_xticks([2000, 4000, 6000])
    # elif 'Fertility' in feature_name[feature_ind]:
    #     axs.set_xticks([-1, -0.5, 0, 0.5, 1])
    # elif 'PAR' in feature_name[feature_ind]:
    #     axs.set_xticks([160, 180, 200, 220])
    # elif 'MCWD' in feature_name[feature_ind]:
    #     axs.set_xticks([-600, -400, -200, 0])

# Add colorbar to the right of the plots
cbar_ax = fig.add_axes([0.125, 0.10, 0.8, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Number of samples', fontsize=7)
cbar.ax.tick_params(labelsize=6)

# for i in range(10, 12):
#     axes.flatten()[i].axis("off")  # 关闭坐标轴

# cax = fig.add_axes([0.550, 0.2, 0.4, 0.03])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cax, label="Color Scale")
# cbar = fig.colorbar(im,  shrink=1.2, fraction=0.08, aspect=30, pad=0.05,orientation='horizontal', )  #, ax=axes, cax=cax,
# cbar.ax.tick_params(labelsize=6)
# cbar.set_label('Number of samples', fontsize=7)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Supplementary_Figures\FigS9_Partial_dependence_plot_4x3_0521.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] Plot the summary plot of the shap value with plot.bar


width_x, height_y = 650, 786

# feature_name = ['PAR (W m$^{-2}$)', 'MCWD (mm)', 'MAP (mm)', 'VPD (kPa)', 'Soil Clay',
#                 'Soil Fertility (ln(cmol(+) kg$^{-1}$))', 'Soil moisture (mm)', 'Soil Sand', 'Canopy Height(m)',
#                 'Isohydricity']  #'DSL (month)',

# feature_name = ['Canopy Height', 'PAR', 'Precipitation', 'Soil Clay', 'Soil Fertility', 'Soil Sand']
# feature_name = ['Canopy Height', 'PAR', 'Precipitation', 'Isohydricity', 'Soil Clay', 'Soil Fertility', 'Soil Moisture',
#                 'Soil Sand', 'VPD']  # 'Temperature',
# feature_name = ['PAR','Precipitation', 'Soil Moisture', 'VPD', 'Soil Clay','Soil Fertility',  'Soil Sand', 'Canopy Height', 'Isohydricity' ]  # 'Temperature',

# feature_name = ['PAR', 'MCWD', 'MAP', 'VPD', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand', 'Canopy Height', 'Isohydricity']

# feature_name = ['PAR', 'MCWD', 'MAP', 'Herbivory', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand']

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand', ]  # 'DSL (month)',

color_list = [
    'tab:orange',  # PAR 0
    'tab:brown',  # VPD 1
    'lightgray',  # Soil 2
    'tab:blue',  # Hydroclimate 3
]
# feature_color = [color_list[0], color_list[1], color_list[1], color_list[1], color_list[2], color_list[2],
#                  color_list[2], color_list[3], color_list[3]]

# 0: PAR, 1: VPD, 2: soil; 3:  Hydroclimate
feature_color = [color_list[0], color_list[3], color_list[3], color_list[1], color_list[2], color_list[2],
                 color_list[3], color_list[2]]

feature_importance = np.nanmean(np.abs(shap_values), axis=0)

# normalize the feature importance
feature_importance = feature_importance / np.sum(np.array(feature_importance))

sorted_idx = feature_importance.argsort()

# set color for each feature

# type_list_id = set(feature_color)
type1 = [feature_name.index(x) for x in feature_name if feature_color[feature_name.index(x)] == color_list[0]]  # PAR
type2 = [feature_name.index(x) for x in feature_name if
         feature_color[feature_name.index(x)] == color_list[3]]  # Hydroclimate
type3 = [feature_name.index(x) for x in feature_name if feature_color[feature_name.index(x)] == color_list[2]]  # soil
type4 = [feature_name.index(x) for x in feature_name if
         feature_color[feature_name.index(x)] == color_list[1]]  # Herbivory

class_importance = np.zeros(4)

class_importance[0] = np.sum(np.array([feature_importance[x] for x in type3]))  # soil
class_importance[1] = np.sum(np.array([feature_importance[x] for x in type2]))  # hydroclimate
class_importance[2] = np.sum(np.array([feature_importance[x] for x in type1]))  # PAR
class_importance[3] = np.sum(np.array([feature_importance[x] for x in type4]))  # VPD

# ensure keep 2 decimal of each value and sum of them equal to 1
class_importance = np.round(class_importance, 3)

if np.sum(class_importance) != 1:
    index_max = np.argmax(class_importance)
    class_importance[index_max] = 1 - np.sum(class_importance) + class_importance[index_max]

# Dominant class across the basin

Full_feature_map = copy.deepcopy(amazon_features)
#
# # five types of drivers
five_types = [type1, type2, type3, type4]
#
dominant_map = np.zeros((width_x, height_y)) * np.nan
#
for id in np.arange(len(five_types)):
    ind_list = five_types[id]
    for ind in ind_list:
        ind_x, ind_y = np.where(Full_feature_map == ind)
        dominant_map[ind_x, ind_y] = id

dominant_map_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_four_class_herbivory_0724.tif'
# save_tif(dominant_map, dominant_map_path, _geo, _prj, 1)

# In[] bar plot of each feature importance

color_list = [
    'tab:orange',  # PAR 0
    'tab:brown',  # VPD 1
    'lightgray',  # Soil 2
    'tab:blue',  # Hydroclimate 4
]

fig, ax1 = plt.subplots(figsize=(6, 8))

# ax1.barh([feature_name[x] for x in sorted_idx], feature_importance[sorted_idx],
#          color=[feature_color[x] for x in sorted_idx], alpha=0.8)
ax1.barh([feature_name[x] for x in sorted_idx], feature_importance[sorted_idx],
         color=[feature_color[x] for x in sorted_idx])  # , alpha=0.8

ax1.tick_params(axis='both', which='major', length=3)

ax1.set_xlabel('Feature Importance', fontsize=16)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.set_yticks([feature_name[x] for x in sorted_idx],fontsize=16)

# left, bottom, width, height = [0.53, 0.10, 0.35, 0.35]
left, bottom, width, height = [0.450, 0.2, 0.45, 0.45]

ax2 = fig.add_axes([left, bottom, width, height])

# Plot the pie chart

ax2.pie(class_importance, labels=['Soil', 'Hydroclimate', 'PAR', 'Herbivory'], pctdistance=0.56,
        labeldistance=1.05,
        autopct='%1.1f%%', colors=[color_list[2], color_list[3], color_list[0], color_list[1]],
        # wedgeprops={"alpha": 0.8},
        textprops={'fontsize': 14, 'color': 'k', 'weight': 'regular'})  #

# plt.legend(fontsize=8, ncol=3, loc='lower right', frameon=False)
plt.tight_layout()
save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Feature_importance_herbivory.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')


# the relative importance of each feature across all the features
for i in sorted_idx:
    feature_importance[i] = feature_importance[i] / np.sum(np.array(feature_importance))
    print(feature_name[i], " is ", feature_importance[i])

# Show the plot
# plt.show()

# In[] Main Fig 3 Overall feature importance and partial dependence plot

feature_name_bar = ['SR', 'MCWD', 'MAP', 'Herbivory', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand', ]

feature_name = ['SR(W/m$^{2}$)', 'MCWD(mm)', 'MAP(mm/year)', 'Herbivory', 'Soil Clay',
                'Soil Fertility(cmol(+)/kg)', 'Soil moisture(m$^3$ m$^{-3}$)', 'Soil Sand', ]  # 'DSL (month)',
# "10-based logarithms"
feature_ind_list = important_ind[:3]  # [:3]

# fig, axs = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
fig, axes = plt.subplots(2, 2, figsize=(6, 5.5),dpi=150)  #

plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.20, wspace=0.30, hspace=0.4)

norm1 = mcolors.Normalize(vmin=0, vmax=1500)
im = cm.ScalarMappable(norm=norm1, cmap='coolwarm')

for tmp_ind in np.arange(4):
    if tmp_ind == 0:
        ax1 = axes.flatten()[tmp_ind]
        ax1.barh([feature_name_bar[x] for x in sorted_idx], feature_importance[sorted_idx],
                 color=[feature_color[x] for x in sorted_idx])  # , alpha=0.8

        ax1.tick_params(axis='both', which='major', length=2)

        ax1.set_xlabel('Feature Importance', fontsize=8)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=7)
        # ax1.set_yticks([feature_name[x] for x in sorted_idx],fontsize=16)

        # left, bottom, width, height = [0.53, 0.10, 0.35, 0.35]
        pie_left, pie_bottom, pie_width, pie_height = [0.38, 0.03, 0.5, 0.5]

        # --- 在第一个子图内部添加饼图 ---
        # 定义饼图的位置和大小（相对于 ax1 的坐标）
        # pie_left = 0.6  # 距离 ax1 左边的 60%
        # pie_bottom = 0.6  # 距离 ax1 底部的 60%
        # pie_width = 0.3  # 宽度占 ax1 的 30%
        # pie_height = 0.3  # 高度占 ax1 的 30%

        # 在 ax1 内部创建新的 Axes 对象
        ax2 = fig.add_axes([
            ax1.get_position().x0 + pie_left * ax1.get_position().width,
            ax1.get_position().y0 + pie_bottom * ax1.get_position().height,
            pie_width * ax1.get_position().width,
            pie_height * ax1.get_position().height
        ])

        # Plot the pie chart
        wedges, texts, autotexts = ax2.pie(
            class_importance,
            labels=['Soil', 'Hydroclimate', 'Light', 'Herbivory'],
            pctdistance=0.56,  # 百分比文本距离圆心位置
            labeldistance=1.05,  # 标签距离圆心位置
            autopct='%1.1f%%',  # 百分比格式
            colors=[color_list[2], color_list[3], color_list[0], color_list[1]],
            textprops={'fontsize': 6, 'weight': 'regular'}  # 初始统一样式
        )
        # # 修改标签颜色为黑色
        # for text in texts:
        #     text.set_color('black')
        #
        # # 修改百分比文本颜色为白色
        # for autotext in autotexts:
        #     autotext.set_color('white')

    else:
        feature_ind = feature_ind_list[tmp_ind - 1]
        input_y = shap_values[:, feature_ind]
        # input_y = copy.deepcopy(data_y)
        input_x = data_x[:, feature_ind]

        # select the data within the range of 5% to 95% percentile
        threshold = 0.05
        p95 = np.nanpercentile(input_x, 100 - threshold)
        p5 = np.nanpercentile(input_x, threshold)

        input_x_mask = np.logical_and(input_x < p95, input_x > p5)

        # if 'PAR' in feature_name[feature_ind]:
        #     input_x_mask = input_x > 160

        fit_x = input_x[input_x_mask]
        fit_y = input_y[input_x_mask]

        # axs[i, j].hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')
        if tmp_ind == 1:
            ax_id = 2
        elif tmp_ind == 2:
            ax_id = 1
        elif tmp_ind == 3:
            ax_id = 3
        axs = axes.flatten()[ax_id]
        axs.hexbin(fit_x, fit_y, gridsize=80, mincnt=1, cmap='coolwarm', alpha=0.5, edgecolors='None')

        fit_degree = 2
        # if 'VPD' in feature_name[feature_ind]:
        #     fit_degree = 3
        #
        # elif 'MAP' in feature_name[feature_ind]:
        #     fit_degree = 2
        #     # fit_x = fit_x * 1000
        # # elif 'PAR' in feature_name[feature_ind]:
        # #     fit_degree = 3
        # else:
        #     fit_degree = 2

        # model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
        # bins = np.linspace(fit_x.min(), fit_x.max(), 200)

        bins = np.linspace(fit_x.min(), fit_x.max(), 200)

        input_x_bins = np.digitize(fit_x, bins)
        #
        bin_statistics = np.array([np.nanmean(fit_y[input_x_bins == i]) for i in range(1, len(bins) + 1)])
        # bin_statistics_std_up = [np.nanpercentile(input_y[input_x_bins == i], 95) for i in range(1, len(bins) + 1)]
        # bin_statistics_std_down = [np.nanpercentile(input_y[input_x_bins == i], 5) for i in range(1, len(bins) + 1)]
        try:
            model_fit = np.poly1d(np.polyfit(fit_x, fit_y, fit_degree))
        except:
            model_fit = np.poly1d(np.polyfit(bins, bin_statistics, fit_degree))  # in case of too much data hide the relationship
            # pass
        # model_fit = np.poly1d(np.polyfit(bins, bin_statistics, fit_degree))  # in case of too much data hide the relationship

        axs.plot(bins, model_fit(bins), 'k', linestyle='-', linewidth=1.0, alpha=0.8)

        # sns.regplot(x=fit_x, y=fit_y,  ci=95, line_kws={'linestyle': '-', 'color': 'k'},
        #             ax=axs[i, j],scatter=False, order=fit_degree)

        # axs[i, j].plot(bins, bin_statistics, 'o', )  # color=feature_color[feature_ind])  # Plot mean deciduousness amplitude against bin edges

        # plt.fill_between(bins, bin_statistics_std_up, bin_statistics_std_down, alpha=0.2, color=feature_color[feature_ind])
        # plt.plot(bins, bin_statistics, 'o', )  # color=feature_color[feature_ind])  # Plot mean deciduousness amplitude against bin edges
        # sns.regplot(x=bins, y=bin_statistics, order=1, ci=90, scatter_kws={'s': 40, 'alpha': 0.6},
        #             line_kws={'linestyle': '--', 'color': 'k'})

        # axs[i, j].set_title(feature_name[feature_ind], fontsize=18)
        axs.set_xlabel(feature_name[feature_ind], fontsize=7)

        # if 'Fertility' in feature_name[feature_ind]:
        #     axs.set_ylabel('SHAP value for Soil fertility', fontsize=7)
        # else:
        #     axs.set_ylabel('SHAP value for {}'.format(feature_name[feature_ind].split(' ')[0]), fontsize=7)
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
        # if feature_name[feature_ind] == 'PAR':
        #     axs[i, j].set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        #     axs[i, j].set_yticklabels([-0.2, -0.1, 0.0, 0.1, 0.2])
        # elif feature_name[feature_ind] == 'Precipitation':
        #     axs[i, j].set_yticks([ -0.1, 0.0, 0.1, 0.2, 0.3])
        #     axs[i, j].set_yticklabels([ -0.1, 0.0, 0.1, 0.2, 0.3])
        # else:
        #     axs[i, j].set_yticks([ -0.1, 0.0, 0.1, 0.2])
        #     axs[i, j].set_yticklabels([ -0.1, 0.0, 0.1, 0.2])
        # if feature_name[feature_ind] == 'VPD':
        #     axs[i, j].set_yticks([ -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
        #     axs[i, j].set_yticklabels([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
        #
        # else:
        #     axs[i, j].set_yticks([  -1.0, 0.0, 1.0, 2.0])
        #     axs[i, j].set_yticklabels([ -1.0, 0.0, 1.0, 2.0])
        axs.set_yticks([-1, 0.0, 1])

        axs.tick_params(axis='both', which='major', labelsize=7)

        axs.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # if 'XXX' in feature_name[feature_ind]:
    #     axs.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # elif 'MAP' in feature_name[feature_ind]:
    #     axs.set_xticks([2000, 4000, 6000])
    # elif 'Fertility' in feature_name[feature_ind]:
    #     axs.set_xticks([-1, -0.5, 0, 0.5, 1])
    # elif 'PAR' in feature_name[feature_ind]:
    #     axs.set_xticks([160, 180, 200, 220])
    # elif 'MCWD' in feature_name[feature_ind]:
    #     axs.set_xticks([-600, -400, -200, 0])
cax = fig.add_axes([0.13, 0.075, 0.8, 0.03])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cax, label="Color Scale")
cbar = fig.colorbar(im, cax=cax, shrink=1.2, fraction=0.08, aspect=30, pad=0.05,
                    orientation='horizontal', )  # , ax=axes,
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Number of samples', fontsize=8)

# plt.tight_layout()

# cbar = fig.colorbar(im, ax=axes.flatten()[2], shrink=1.2, fraction=0.08, aspect=30, pad=0.05, orientation='horizontal')  # ,
# cbar.ax.tick_params(labelsize=8)
# cbar.set_label('Number of samples', fontsize=9)

# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Fig4_driver_Vegetation_removal_0521.png'
save_path = r'/Users/song/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/PhD_Projects/Project4_Mapping_Amazon_Basin_Using_GEE/Manuscript/Python_output_figures/Fig4_driver_Vegetation_removal_0808.png'
# plt.savefig(save_path, dpi=300)

# In[] save shap value of top 3 types of drivers

width_x, height_y = 650, 786

abs_shap_vaule = np.abs(shap_values)

type1_shap = abs_shap_vaule[:, type1].reshape(-1)  # PAR

type2_shap = np.nansum(abs_shap_vaule[:, type2], axis=1)  # Hydroclimate

type3_shap = np.nansum(abs_shap_vaule[:, type3], axis=1)  # soil

type4_shap = np.nansum(abs_shap_vaule[:, type4], axis=1)  # vegetation

# drivers_4map = np.zeros((width_x, height_y, 4)) * np.nan
#
# drivers_4map_res = drivers_4map.reshape(-1, 4)
#
# drivers_4map_res[~final_mask, 0] = type1_shap  # par
# drivers_4map_res[~final_mask, 1] = type2_shap  # hydroclimate
# drivers_4map_res[~final_mask, 2] = type3_shap  # soil
# drivers_4map_res[~final_mask, 3] = type4_shap  # vegetation
#
# drivers_dom = np.argmax(drivers_4map_res, axis=1)
#
# drivers_dom[final_mask] = 4
# drivers_dom_map = drivers_dom.reshape(width_x, height_y)
#
# # drivers_4map = drivers_4map_res.reshape(drivers_4map.shape)
# # drivers_dominant = np.argmax(drivers_4map, axis=2)
#
# # save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\drivers_dominant.tif'
# # save_tif(drivers_dom_map, save_path, _geo, _prj, 1)
#
drivers_top3_map = np.zeros((width_x, height_y, 3)) * np.nan
#
drivers_top3_map_res = drivers_top3_map.reshape(-1, 3)
#
drivers_top3_map_res[~final_mask, 0] = type1_shap # Light
drivers_top3_map_res[~final_mask, 1] = type2_shap # Hydroclimate
drivers_top3_map_res[~final_mask, 2] = type3_shap # Soil
#
drivers_top3_map = drivers_top3_map_res.reshape(drivers_top3_map.shape)
#
drivers_top3_map_nor = drivers_top3_map / np.nansum(drivers_top3_map, axis=2)[:, :, np.newaxis]

save_path = r'/Users/song/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/PhD_Projects/Project4_Mapping_Amazon_Basin_Using_GEE/Manuscript/Python_output_figures/Tif/Asynchrony_shap_map_3type_drivers_0801.tif'

save_tif(drivers_top3_map_nor, save_path, _geo, _prj, 3)

# save_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\Feature_importance\Asynchrony_shap_map_3type_drivers_1111.tif'
# save_tif(drivers_top3_map_nor, save_path, _geo, _prj, 3)

# drivers_top3 = np.stack(type1_shap, type2_shap, type3_shap)

# shap_values_all = np.nanmean(np.abs(shap_values), axis=0)

# np.sum(feature_importance[sorted_idx][:4]) / np.sum(feature_importance)
# np.sum(np.array([feature_importance[x] for x in important_ind[:3]])) / np.sum(feature_importance)

# plt.figure(figsize=(10, 6))
# # plt.barh([feature_name[x] for x in sorted_idx], feature_importance[sorted_idx])
# plt.barh([feature_name[x] for x in sorted_idx], feature_importance[sorted_idx], color=[feature_color[x] for x in sorted_idx], alpha=0.8)
# plt.xlabel('Feature Importance', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
#
# # plot the pie chart for the feature importance; sum proportion of the feature importance with the same color
# plt.figure(figsize=(6, 6))
# plt.pie(class_importance, labels=['Canopy Height','Climate Variables', 'Soil Properties'], labeldistance=0.3, pctdistance=0.5,
#         autopct='%1.2f%%', colors=[color_list[2], color_list[0], color_list[1]], wedgeprops={"alpha": 0.8},textprops={'fontsize': 14})
# # plt.legend(fontsize=12,ncol=3, loc='lower center', frameon=False)
#
# # plt.xlabel('Feature Importance', fontsize=18)
# plt.tight_layout()


# In[] Cope with WTD data
# wtd_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Climate_Variables\amazon_wtd.tif'
#
# _geo,_prj,wtd = readTif_gdal(wtd_path)
#
# wtd_mB = cv2.medianBlur(wtd, 5)
#
# wtdMB_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Climate_Variables\amazon_wtd_mB.tif'
# wtd_mB[wtd_mB<-1000]=np.nan
# save_tif(wtd_mB,wtdMB_path,_geo,_prj,1)
#
# wtd_mB_coarse = cv2.resize(wtd_mB, (raws_y, columns_x))
#
# wtd_mB_coarse[~forest_mask]=np.nan
#
# wtdMB_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Climate_Variables\amazon_wtd_mB_forest.tif'
#
# save_tif(-wtd_mB_coarse,wtdMB_path,_geo,_prj,1)

# radiation_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\climate_par.tif'
# _geo, _prj, radiation = readTif_gdal(radiation_path)
#
# radiation_mean = np.nanmean(radiation, axis=2)
#
# save_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\climate_par_mean.tif'
#
# save_tif(radiation_mean, save_path, _geo, _prj, 1)
#
#
# pre_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\climate_precipitation.tif'
# _geo, _prj, precipitation = readTif_gdal(pre_path)
#
# precipitation_sum = np.nansum(precipitation, axis=2)
#
# save_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\climate_precipitation_sum.tif'
#
# save_tif(precipitation_sum, save_path, _geo, _prj, 1)

# soil_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\raw\soil_moisture.tif'
#
# _geo,_prj,soil_moisture = readTif_gdal(soil_path)
#
# soil_moisture_mean = np.nanmean(soil_moisture, axis=2)
#
# save_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\soil_moisture_mean_terr.tif'
# save_tif(soil_moisture_mean, save_path, _geo, _prj, 1)

# In[] Extract the dry season length from the precipitation data

# pre_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\raw\precipitation.tif'
# _geo, _prj, precipitation = readTif_gdal(pre_path)
#
# nan_mask = np.sum(np.isnan(precipitation),axis=2)==12
# dry_season = precipitation < 100
#
# dry_season_length = np.nansum(dry_season, axis=2)
# dry_season_length = dry_season_length.astype(np.float32)
# dry_season_length[nan_mask] = np.nan
#
# save_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\raw\DSL.tif'
# save_tif(dry_season_length, save_path, _geo, _prj, 1)

# In[] extract the MCWD Amazon from 2019 to 2021

# year_range = np.arange(2001,2024)
#
# mcwd_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\raw\Amazon_MCWD_2001_20230.TIF'
#
# _geo, _prj, mcwd = readTif_gdal(mcwd_path)
#
# mcwd = mcwd.transpose(1,2,0)
#
# mcwd_2019_2021 = mcwd[:,:,-5:-2]
#
# mcwd_mean = np.nanmean(mcwd_2019_2021, axis=2)
#
# # save_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\raw\Amazon_MCWD_mean_3yr.tif'
# # save_tif(mcwd_mean, save_path, _geo, _prj, 1)
#
#
# # MCWD_3year = np.zeros((mcwd.shape[0],mcwd.shape[1],3))
#
# A = mcwd_mean < -1000

# In[] Extract the residual of the gs from VPD

gs_dir = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\stomata'
gs_list = [x for x in os.listdir(gs_dir) if x.endswith('.tif')]

monthly_indices = [
    [0, 3],  # 1月
    [3, 6],  # 2月
    [6, 9],  # 3月
    [9, 12],  # 4月
    [12, 15],  # 5月
    [15, 18],  # 6月
    [18, 21],  # 7月
    [21, 24],  # 8月
    [24, 27],  # 9月
    [27, 30],  # 10月
    [30, 33],  # 11月
    [33, 46]  # 12月
]

gs_list.sort()

gs_month_list = []
# extract the mean value of the gs
for gs_name in gs_list:
    gs_file_path = os.path.join(gs_dir, gs_name)
    _geo, _prj, gs = readTif_gdal(gs_file_path)
    monthly_mean_data = np.zeros((12, 520, 820))

    for i in range(12):
        monthly_mean_data[i] = np.nanmean(gs[monthly_indices[i][0]:monthly_indices[i][1], :, :], axis=0)

    gs_month_list.append(monthly_mean_data)

gs_mean = np.nanmean(np.stack(gs_month_list), axis=0)

gs_mean = gs_mean.transpose(1, 2, 0)

gs_mean_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\stomata\Mean\gs_mean_3yr.tif'

save_tif(gs_mean, gs_mean_path, _geo, _prj, 12)
# In[]

gs_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\stomata\Mean\gs_mean_3yr_Amazon0.TIF'
vpd_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\hydroclimate_vpd_Amazon.tif'

_geo, _prj, gs_mean = readTif_gdal(gs_path)
_geo, _prj, vpd = readTif_gdal(vpd_path)

gs_mean = cv2.resize(gs_mean, (vpd.shape[1], vpd.shape[0]))

gs_mean[gs_mean < -1] = np.nan

gs_mean2 = np.nanmean(gs_mean, axis=2)

vpd_mean = np.nanmean(vpd, axis=2)

df = pd.DataFrame({'gs': gs_mean2.flatten(), 'VPD': vpd_mean.flatten()})

df = df.dropna()

corr = df[['gs', 'VPD']].corr().iloc[0, 1]
print(f"Pearson R (gs vs. VPD): {corr:.2f}")

vif = variance_inflation_factor(df.values, 0)
