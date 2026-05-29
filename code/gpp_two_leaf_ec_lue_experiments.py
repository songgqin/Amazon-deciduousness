"""Two-leaf EC-LUE GPP experiment runner.

"""

# In[] Imports
import copy
import re
import os
import numpy as np
import cv2
import pandas as pd
from osgeo import gdal
import matplotlib

# In[] Workflow
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 150

# In[] Functions
def readTif_gdal(fileName, nbands=36):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)

    if dataset is None:
        raise FileNotFoundError(f"Cannot open file: {fileName}")

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    if im_data.ndim == 3 and im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])

    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif(array, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()

    outputData = driver.Create(
        savePath,
        array.shape[1],
        array.shape[0],
        nbands,
        gdal.GDT_Float32
    )

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(array.astype(np.float32))
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(array[:, :, i].astype(np.float32))
            outputData.GetRasterBand(i + 1).SetNoDataValue(np.nan)

    del outputData

def resize_to_target(arr, target_width, target_height):
    return cv2.resize(arr, (target_width, target_height))

def calculate_Cs(Ca, phi, n, Po, Ta, R, VPD):
    Kc = 39.97 * np.exp((79.43 * (Ta - 298.15)) / (298.15 * Ta * R))
    Ko = 27480 * np.exp((36.38 * (Ta - 298.15)) / (298.15 * Ta * R))

    K = Kc * (1.0 + Po / Ko)
    gamma = np.sqrt((356.51 * K) / (1.6 * n))

    chi = gamma / (gamma + np.sqrt(VPD * 1000.0))

    Ci = Ca * chi
    Cs = (Ci - phi) / (Ci + 2.0 * phi)

    return Cs.astype(np.float32)

def generate_lat_grid(geotransform, rows, cols):
    lat_top_left = geotransform[3]
    lat_res = geotransform[5]

    lon_top_left = geotransform[0]
    lon_res = geotransform[1]

    lat_1d = lat_top_left + (np.arange(rows) + 0.5) * lat_res
    lon_1d = lon_top_left + (np.arange(cols) + 0.5) * lon_res

    _, lat_grid = np.meshgrid(lon_1d, lat_1d)

    return lat_grid.astype(np.float32)

def calculate_monthly_basin_sza(lat_grid):
    doy_map = {
        1: 15, 2: 46, 3: 75, 4: 105,
        5: 135, 6: 166, 7: 196, 8: 227,
        9: 258, 10: 288, 11: 319, 12: 349
    }

    month_array = np.arange(1, 13)
    doy_array = np.array([doy_map[m] for m in month_array])

    doy = doy_array[None, None, :]
    lat_rad = np.radians(lat_grid)[:, :, None]

    delta = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

    tan_lat_delta = -np.tan(lat_rad) * np.tan(delta)
    tan_lat_delta = np.clip(tan_lat_delta, -1.0, 1.0)

    omega_s = np.arccos(tan_lat_delta)
    omega_s = np.clip(omega_s, 1e-6, None)

    cos_theta_mean = (
        np.sin(lat_rad) * np.sin(delta) +
        (np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)) / omega_s
    )

    cos_theta_mean = np.clip(cos_theta_mean, 0.05, 1.0)

    SZA_mean = np.degrees(np.arccos(cos_theta_mean))

    return SZA_mean.astype(np.float32)

def calculate_two_leaf_APAR(LAI_grid, PAR_dir_grid, PAR_dif_grid, theta_deg_grid, Omega=0.8):
    theta = np.radians(theta_deg_grid)
    beta = np.radians(60.0)
    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, 1e-6, None)

    LAI_su = 2.0 * cos_theta * (1.0- np.exp(-0.5 * Omega * LAI_grid / cos_theta))
    LAI_su = np.clip(LAI_su, 0.0, LAI_grid)
    LAI_sh = LAI_grid - LAI_su

    cos_szaa = 0.537 + 0.025 * LAI_grid
    cos_szaa = np.clip(cos_szaa, 1e-6, None)

    PARdif_under = PAR_dif_grid * np.exp(-0.5 * Omega * LAI_grid / cos_szaa)

    lai_factor = 1.1 - 0.1 * LAI_grid
    C = 0.07 * Omega * PAR_dir_grid * lai_factor * np.exp(-cos_theta)

    diffuse_term = np.divide(
        PAR_dif_grid - PARdif_under,
        LAI_grid,
        out=np.zeros_like(LAI_grid, dtype=np.float32),
        where=(LAI_grid > 1e-6)
    )

    APAR_su = (
        PAR_dir_grid * (np.cos(beta) / cos_theta) +
        diffuse_term +
        C
    ) * LAI_su

    APAR_sh = (diffuse_term + C) * LAI_sh

    return APAR_su.astype(np.float32), APAR_sh.astype(np.float32)

target_width, target_height = 786, 650

T_path = r'data/climate/Temp2019_2021_ERA.tif'
VPD_path = r'data/climate/VPD_Amazon.tif'
LAI_path = r'data/gpp/inputs/Amazon_MOD15_LAI.tif'

Par_dir_path = r'data/climate/ERA5_PAR_dir_MJ_2019_2021_Avg_12bands.tif'
Par_dif_path = r'data/climate/ERA5_PAR_dif_MJ_2019_2021_Avg_12bands.tif'

leaf_age_path = r'data/leaf_age/Leaf_Age_ln_Dec_Litterfall_LAI_0414v2.tif'

dec_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'

param_summary_path = r'data/gpp/parameters/Two_Leaves_EC_LUE_Models_Parameters_Opt_Summary.csv'

out_mod_lai_path = r'data/gpp/outputs/TL_EC_LUE_GPP_local.tif'
out_dec_lai_path = r'data/gpp/outputs/TL_EC_LUE_LAI_Dec_GPP.tif'
out_ld_age_path = r'data/gpp/outputs/TL_EC_LUE_LAI_Dec_Demography_GPP.tif'

dec_geo, dec_prj, dec_data = readTif_gdal(dec_path)
dec_data = resize_to_target(dec_data, target_width, target_height)
dec_data = np.array(dec_data, dtype=np.float32)
dec_data[dec_data > 1000] = np.nan
dec_data_nor = dec_data / 1000.0

_, _, T_k = readTif_gdal(T_path)
_, _, VPD = readTif_gdal(VPD_path)
_, _, LAI = readTif_gdal(LAI_path)
_, _, Par_dir = readTif_gdal(Par_dir_path)
_, _, Par_dif = readTif_gdal(Par_dif_path)
_, _, LA = readTif_gdal(leaf_age_path)

VPD[VPD < 0] = np.nan

T_k = resize_to_target(T_k, target_width, target_height).astype(np.float32)
T = T_k - 273.15

VPD = VPD * 0.1
VPD = resize_to_target(VPD, target_width, target_height).astype(np.float32)

Par_dir = resize_to_target(Par_dir, target_width, target_height).astype(np.float32)
Par_dif = resize_to_target(Par_dif, target_width, target_height).astype(np.float32)

LAI = 0.1 * LAI
LAI = resize_to_target(LAI, target_width, target_height).astype(np.float32)

LA = resize_to_target(LA, target_width, target_height).astype(np.float32)

LAIY = LA[:, :, ::3]
LAIM = LA[:, :, 1::3]
LAIO = LA[:, :, 2::3]

LAI_sum = LAIY + LAIM + LAIO

LAIY_nor = np.divide(
    LAIY,
    LAI_sum,
    out=np.zeros_like(LAIY, dtype=np.float32),
    where=(LAI_sum > 1e-6)
)

LAIM_nor = np.divide(
    LAIM,
    LAI_sum,
    out=np.zeros_like(LAIM, dtype=np.float32),
    where=(LAI_sum > 1e-6)
)

LAIO_nor = np.divide(
    LAIO,
    LAI_sum,
    out=np.zeros_like(LAIO, dtype=np.float32),
    where=(LAI_sum > 1e-6)
)

ratio_Y, ratio_M, ratio_O = 0.75, 1.0, 0.50

age_factor = (
    ratio_Y * LAIY_nor +
    ratio_M * LAIM_nor +
    ratio_O * LAIO_nor
)

mean_age_factor = np.nanmean(age_factor,axis=2)

Ca_2019 = [409.92, 410.34, 410.89, 411.33, 411.34, 410.53, 408.88, 407.64, 407.92, 409.44, 410.87, 411.76]
Ca_2020 = [412.43, 412.95, 413.44, 413.86, 413.81, 412.88, 411.17, 409.73, 410.00, 411.66, 413.25, 414.14]
Ca_2021 = [414.74, 415.20, 415.49, 415.81, 416.01, 415.20, 413.45, 412.15, 412.38, 413.83, 415.58, 416.60]

Ca = (np.array(Ca_2019) + np.array(Ca_2020) + np.array(Ca_2021)) / 3

phi = 20
n = 0.8891
Po = 21300.0
R = 8.314

Tmax = 48.0
Tmin = 2.0
Topt = 28.0

Cs = calculate_Cs(Ca, phi, n, Po, T_k, R, VPD)

numerator = (T - Tmax) * (T - Tmin)
denominator = numerator - (T - Topt) * (T - Topt)
denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)

Ts = numerator / denominator
Ts = np.clip(Ts, 0.0, 1.0).astype(np.float32)

lat_grid = generate_lat_grid(dec_geo, target_height, target_width)
SZA_mean = calculate_monthly_basin_sza(lat_grid)

param_df = pd.read_csv(param_summary_path, encoding='utf-8-sig')

def parse_param_from_text(param_text, param_name):
    pattern = rf'{param_name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    match = re.search(pattern, str(param_text))

    if match is None:
        raise ValueError(f"Cannot find {param_name} in parameter text: {param_text}")

    return float(match.group(1))

def get_tl_lue_params(param_df, model_name):
    row = param_df.loc[param_df['Model'] == model_name]

    if len(row) == 0:
        raise ValueError(f"Model {model_name} not found in parameter CSV.")

    row = row.iloc[0]

    if all(col in param_df.columns for col in ['Emsu_median', 'Emsh_median', 'VPD0_median']):
        Emsu = float(row['Emsu_median'])
        Emsh = float(row['Emsh_median'])
        VPD0 = float(row['VPD0_median'])
    else:
        param_text = row['Optimized Parameters (Median with 95% CI)']
        Emsu = parse_param_from_text(param_text, 'Emsu')
        Emsh = parse_param_from_text(param_text, 'Emsh')
        VPD0 = parse_param_from_text(param_text, 'VPD0')

    return Emsu, Emsh, VPD0

def calc_ws_ec(VPD, VPD0):
    Ws = VPD0 / (VPD + VPD0)
    Ws = np.clip(Ws, 0.0, 1.0)
    return Ws.astype(np.float32)

def get_lai_input(input_type, LAI, dec_data_nor):
    if input_type == 'MOD_LAI':
        return LAI.astype(np.float32)

    if input_type in ['Dec_LAI', 'LD_Age']:
        LAI_est = np.clip(8.241 * (1.0 - dec_data_nor) - 1.991, 0.0, None)
        return LAI_est.astype(np.float32)

    raise ValueError(f"Unknown input_type: {input_type}")

def calc_tl_lue_gpp(
    input_type,
    Emsu,
    Emsh,
    VPD0,
    LAI,
    dec_data_nor,
    Par_dir,
    Par_dif,
    SZA_mean,
    Ts,
    VPD,
    LAIY_nor=None,
    LAIM_nor=None,
    LAIO_nor=None,
    ratio_Y=0.75,
    ratio_M=1.0,
    ratio_O=0.5,
    use_cs=False,
    Cs=None
):
    LAI_input = get_lai_input(
        input_type=input_type,
        LAI=LAI,
        dec_data_nor=dec_data_nor
    )

    APAR_su, APAR_sh = calculate_two_leaf_APAR(
        LAI_input,
        Par_dir,
        Par_dif,
        SZA_mean
    )

    Ws = calc_ws_ec(VPD, VPD0)
    stress = np.minimum(Ts, Ws)

    if input_type == 'LD_Age':
        if LAIY_nor is None or LAIM_nor is None or LAIO_nor is None:
            raise ValueError("LD_Age requires LAIY_nor, LAIM_nor and LAIO_nor.")

        Esum_su = (
            Emsu * ratio_Y * LAIY_nor +
            Emsu * ratio_M * LAIM_nor +
            Emsu * ratio_O * LAIO_nor
        )

        Esum_sh = (
            Emsh * ratio_Y * LAIY_nor +
            Emsh * ratio_M * LAIM_nor +
            Emsh * ratio_O * LAIO_nor
        )

        GPP = (APAR_su * Esum_su + APAR_sh * Esum_sh) * stress

    else:
        if len(Emsu.shape) == 2:
            Emsu = Emsu[:, :, np.newaxis]
            Emsh = Emsh[:, :, np.newaxis]

        GPP = (Emsu * APAR_su + Emsh * APAR_sh) * stress

    return GPP.astype(np.float32)

PARAM_SOURCE_MODEL = 'TL_LUE_LD_Age'

Emsu_mature, Emsh_mature, VPD0_fixed = get_tl_lue_params(
    param_df,
    PARAM_SOURCE_MODEL
)

Emsu_mean_leaf = Emsu_mature * mean_age_factor
Emsh_mean_leaf = Emsh_mature * mean_age_factor

print("=" * 80)
print("Using fixed TL-LUE LD-framework parameters for all experiments")
print(f"Parameter source : {PARAM_SOURCE_MODEL}")
print(f"Emsu_mature      = {Emsu_mature:.4f}")
print(f"Emsh_mature      = {Emsh_mature:.4f}")

print(f"VPD0_fixed       = {VPD0_fixed:.4f}")
print("=" * 80)

gpp_configs = [
    {
        'output_name': 'TL_LUE_MOD_LAI',
        'input_type': 'MOD_LAI',
        'save_path': out_mod_lai_path
    },
    {
        'output_name': 'TL_LUE_Dec_LAI',
        'input_type': 'Dec_LAI',
        'save_path': out_dec_lai_path
    },
    {
        'output_name': 'TL_LUE_LD_Age',
        'input_type': 'LD_Age',
        'save_path': out_ld_age_path
    }
]

for cfg in gpp_configs:
    output_name = cfg['output_name']
    input_type = cfg['input_type']

    print("=" * 80)
    print(f"Generating {output_name}")

    if input_type == 'LD_Age':

        print("Using dynamic leaf-age capacity")
        print(f"Emsu_mature = {Emsu_mature:.4f}")
        print(f"Emsh_mature = {Emsh_mature:.4f}")
        print(f"VPD0        = {VPD0_fixed:.4f}")

        GPP = calc_tl_lue_gpp(
            input_type=input_type,
            Emsu=Emsu_mature,
            Emsh=Emsh_mature,
            VPD0=VPD0_fixed,
            LAI=LAI,
            dec_data_nor=dec_data_nor,
            Par_dir=Par_dir,
            Par_dif=Par_dif,
            SZA_mean=SZA_mean,
            Ts=Ts,
            VPD=VPD,
            LAIY_nor=LAIY_nor,
            LAIM_nor=LAIM_nor,
            LAIO_nor=LAIO_nor,
            ratio_Y=ratio_Y,
            ratio_M=ratio_M,
            ratio_O=ratio_O,
            use_cs=False,
            Cs=Cs
        )

    else:

        print("Using mean leaf-age capacity")

        print(f"VPD0           = {VPD0_fixed:.4f}")

        GPP = calc_tl_lue_gpp(
            input_type=input_type,
            Emsu=Emsu_mean_leaf,
            Emsh=Emsh_mean_leaf,
            VPD0=VPD0_fixed,
            LAI=LAI,
            dec_data_nor=dec_data_nor,
            Par_dir=Par_dir,
            Par_dif=Par_dif,
            SZA_mean=SZA_mean,
            Ts=Ts,
            VPD=VPD,
            LAIY_nor=LAIY_nor,
            LAIM_nor=LAIM_nor,
            LAIO_nor=LAIO_nor,
            ratio_Y=ratio_Y,
            ratio_M=ratio_M,
            ratio_O=ratio_O,
            use_cs=False,
            Cs=Cs
        )

    save_tif(
        GPP,
        cfg['save_path'],
        dec_geo,
        dec_prj,
        12
    )

    print(f"Saved: {cfg['save_path']}")
    print(f"GPP mean = {np.nanmean(GPP):.3f}")
    print(f"GPP min  = {np.nanmin(GPP):.3f}")
    print(f"GPP max  = {np.nanmax(GPP):.3f}")

print("\nAll TL-EC-LUE GPP maps have been generated using fixed LD-framework parameters.")
