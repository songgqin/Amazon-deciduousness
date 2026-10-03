"""EC-LUE GPP experiment runner.

"""

# In[] Imports
import os
import re
import copy
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal

from sg_smooth import sgfilter_line

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

    im_data = im_data.astype(np.float32)

    im_data[im_data == 65535] = np.nan
    im_data[im_data == -9999] = np.nan

    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()

    datatype = gdal.GDT_Float32

    outputData = driver.Create(
        savePath,
        grouthTif.shape[1],
        grouthTif.shape[0],
        nbands,
        datatype
    )

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(i + 1).SetNoDataValue(np.nan)

    del outputData

def resize_to_target(arr, target_width, target_height, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(arr, (target_width, target_height), interpolation=interpolation)

def auto_scale_vi(x, vi_name):
    x = x.astype(np.float32)
    p99 = np.nanpercentile(x, 99)

    if p99 > 2:
        print(f"[Scale check] {vi_name}: p99={p99:.2f}, divided by 10000.")
        x = x / 10000.0
    else:
        print(f"[Scale check] {vi_name}: p99={p99:.3f}, no scaling applied.")

    return x.astype(np.float32)

def calculate_Cs(Ca, phi, n, Po, Ta, R, VPD):
    Kc = 39.97 * np.exp((79.43 * (Ta - 298.15)) / (298.15 * Ta * R))
    Ko = 27480 * np.exp((36.38 * (Ta - 298.15)) / (298.15 * Ta * R))

    K = Kc * (1 + Po / Ko)
    gamma = np.sqrt((356.51 * K) / (1.6 * n))

    chi = gamma / (gamma + np.sqrt(VPD * 1000))

    Ci = Ca * chi
    Cs = (Ci - phi) / (Ci + 2 * phi)

    return Cs.astype(np.float32)

def gapfill_vi_3cycle_sg(vi_data, min_valid=6, clip_min=0.0, clip_max=1.0):
    vi_data = vi_data.astype(np.float32)

    valid_mask = np.sum(np.isfinite(vi_data), axis=1) >= min_valid
    res = np.full_like(vi_data, np.nan, dtype=np.float32)

    if np.sum(valid_mask) == 0:
        return res

    valid_data = vi_data[valid_mask]

    data_36 = np.concatenate([valid_data, valid_data, valid_data], axis=1)

    df_interp = (
        pd.DataFrame(data_36.T)
        .interpolate(method='linear', limit_direction='both')
        .to_numpy()
        .T
    )

    sg_smoothed_raw = sgfilter_line(
        df_interp.T,
        m1=4,
        d1=2,
        m2=4,
        d2=6,
        sudden_ratio=0.25
    )[0]

    sg_smoothed = sg_smoothed_raw.T

    if sg_smoothed.shape[1] != 36:
        raise ValueError(f"Unexpected SG output shape: {sg_smoothed.shape}")

    middle_12 = sg_smoothed[:, 12:24]

    if clip_min is not None and clip_max is not None:
        middle_12 = np.clip(middle_12, clip_min, clip_max)

    res[valid_mask] = middle_12.astype(np.float32)

    return res

raws_y, columns_x = 786, 650
MIN_VALID_MONTHS = 6
NBANDS_OUT = 12

NDVI_path = r'data/gpp/inputs/BRDF_NDVI.tif'
EVI_path = r'data/gpp/inputs/GPP_BRDF_EVI.tif'
kNDVI_path = r'data/gpp/inputs/BRDF_kNDVI_5km_tanh.tif'

T_path = r'data/climate/Temp2019_2021_ERA.tif'
Par_path = r'data/climate/ERA5_PAR_total_MJ_2019_2021_Avg_12bands.tif'
VPD_path = r'data/climate/VPD_Amazon.tif'
LAI_path = r'data/gpp/inputs/Amazon_MOD15_LAI.tif'

dec_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'
leaf_age_path = r'data/leaf_age/Leaf_Age_ln_Dec_Litterfall_LAI_0414v2.tif'

param_summary_path = r'data/gpp/parameters/EC_LUE_Models_Parameters_Optimization_Summary.csv'

cls_modis_path = r'data/forest_mask/MCD12Q1_Amazon.tif'

print(">>> Reading forest mask...")

_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_resized = resize_to_target(
    cls_md.astype(np.float32),
    raws_y,
    columns_x,
    interpolation=cv2.INTER_NEAREST
)

forest_mask = cls_md_resized == 2

print(f"Forest mask shape: {forest_mask.shape}")
print(f"Forest pixels: {np.sum(forest_mask):,}")

print(">>> Reading basic raster data...")

dec_geo, dec_prj, dec_data = readTif_gdal(dec_path)

dec_data = resize_to_target(dec_data, raws_y, columns_x).astype(np.float32)
dec_data[dec_data > 1000] = np.nan
dec_data = dec_data / 1000.0
dec_data_nor = copy.deepcopy(dec_data)

EVI_Geo, EVI_Proj, EVI = readTif_gdal(EVI_path)
NDVI_geo, NDVI_prj, NDVI = readTif_gdal(NDVI_path)
kNDVI_geo, kNDVI_prj, kNDVI = readTif_gdal(kNDVI_path)

EVI = resize_to_target(EVI, raws_y, columns_x).astype(np.float32)
NDVI = resize_to_target(NDVI, raws_y, columns_x).astype(np.float32)
kNDVI = resize_to_target(kNDVI, raws_y, columns_x).astype(np.float32)

EVI = auto_scale_vi(EVI, 'EVI')
NDVI = auto_scale_vi(NDVI, 'NDVI')
kNDVI = auto_scale_vi(kNDVI, 'kNDVI')

T_Geo, T_Proj, T_k = readTif_gdal(T_path)
Par_Geo, Par_Proj, Par = readTif_gdal(Par_path)
VPD_Geo, VPD_Proj, VPD = readTif_gdal(VPD_path)

T_k = resize_to_target(T_k, raws_y, columns_x).astype(np.float32)
T = T_k - 273.15

Par = resize_to_target(Par, raws_y, columns_x).astype(np.float32)

VPD = VPD * 0.1
VPD = resize_to_target(VPD, raws_y, columns_x).astype(np.float32)
VPD[VPD < 0] = np.nan

LAI_Geo, LAI_Proj, LAI = readTif_gdal(LAI_path)
LAI = 0.1 * LAI
MODIS_LAI = resize_to_target(LAI, raws_y, columns_x).astype(np.float32)
MODIS_LAI[MODIS_LAI < 0] = np.nan

print("Basic data loaded.")

print(">>> Reading leaf-age data...")

LA_Geo, LA_Proj, LA = readTif_gdal(leaf_age_path)

LA = resize_to_target(LA, raws_y, columns_x).astype(np.float32)

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

print("Leaf-age data loaded.")

print(">>> Calculating environmental scalars...")

Ca_2019 = [409.92, 410.34, 410.89, 411.33, 411.34, 410.53, 408.88, 407.64, 407.92, 409.44, 410.87, 411.76]
Ca_2020 = [412.43, 412.95, 413.44, 413.86, 413.81, 412.88, 411.17, 409.73, 410.00, 411.66, 413.25, 414.14]
Ca_2021 = [414.74, 415.20, 415.49, 415.81, 416.01, 415.20, 413.45, 412.15, 412.38, 413.83, 415.58, 416.60]

Ca = (np.array(Ca_2019) + np.array(Ca_2020) + np.array(Ca_2021)) / 3

phi = 30
n = 0.8891
Po = 21300.0
R = 8.314

Tmax = 48
Tmin = 2
Topt = 28

Cs = calculate_Cs(Ca, phi, n, Po, T_k, R, VPD)

numerator = (T - Tmax) * (T - Tmin)
denominator = numerator - (T - Topt) * (T - Topt)
denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)

Ts = numerator / denominator
Ts = np.clip(Ts, 0.0, 1.0).astype(np.float32)

print("Environmental scalars calculated.")

print(">>> Reading EC-LUE parameter summary...")

param_df = pd.read_csv(param_summary_path, encoding='utf-8-sig')

def parse_param_from_text(param_text, param_name):
    pattern = rf'{param_name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    match = re.search(pattern, str(param_text))

    if match is None:
        raise ValueError(f"Cannot find parameter {param_name} in: {param_text}")

    return float(match.group(1))

def get_ec_lue_params(param_df, model_name):
    row = param_df.loc[param_df['Model'] == model_name]

    if len(row) == 0:
        raise ValueError(f"Model {model_name} not found in parameter CSV.")

    row = row.iloc[0]

    if 'Emax_median' in param_df.columns and 'VPD0_median' in param_df.columns:
        Emax = float(row['Emax_median'])
        VPD0 = float(row['VPD0_median'])
    else:
        param_text = row['Optimized Parameters (Median with 95% CI)']
        Emax = parse_param_from_text(param_text, 'Emax')
        VPD0 = parse_param_from_text(param_text, 'VPD0')

    return Emax, VPD0

PARAM_SOURCE_MODEL = 'EC_LUE_LD_Age'
ratio_Y, ratio_M, ratio_O = 0.75, 1.0, 0.50

Emax_fixed, VPD0_fixed = get_ec_lue_params(param_df, PARAM_SOURCE_MODEL)
Emax_mature = Emax_fixed

age_factor = (
    ratio_Y * LAIY_nor +
    ratio_M * LAIM_nor +
    ratio_O * LAIO_nor
)

mean_age_factor = np.nanmean(age_factor,axis=2)

Emax_mean_leaf = Emax_mature * mean_age_factor

print("=" * 80)
print("Using fixed LD-LUE parameters for all EC-LUE experiments")
print(f"Parameter source: {PARAM_SOURCE_MODEL}")
print(f"Emax_mature = {Emax_fixed:.4f}")
print(f"VPD0        = {VPD0_fixed:.4f}")
print("=" * 80)

VI_FPAR_COEF = {
    'NDVI': {'a': 1.24, 'b': -0.168},
    'EVI': {'a': 1.27, 'b': 0.218},
    'kNDVI': {'a': 1.33, 'b': -0.098}
}

FPAR_MIN = 0.0
FPAR_MAX = 0.95

def calc_ws_ec(VPD, VPD0):
    Ws_EC = VPD0 / (VPD + VPD0)
    Ws_EC = np.clip(Ws_EC, 0.0, 1.0)
    return Ws_EC.astype(np.float32)

def build_fpar(input_data, input_type, vi_name=None):
    if input_type == 'VI':
        if vi_name is None:
            raise ValueError("vi_name must be provided when input_type == 'VI'.")

        if vi_name not in VI_FPAR_COEF:
            raise ValueError(f"Unknown vi_name: {vi_name}")

        a_vi = VI_FPAR_COEF[vi_name]['a']
        b_vi = VI_FPAR_COEF[vi_name]['b']

        fpar = a_vi * input_data + b_vi

    elif input_type == 'MOD_LAI':
        fpar = 0.95 - np.exp(-0.5 * input_data)

    elif input_type == 'Dec_LAI':
        LAI_est = np.clip(8.241 * (1.0 - input_data) - 1.991, 0.0, None)
        fpar = 0.95- np.exp(-0.5 * LAI_est)

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    fpar = np.clip(fpar, 0.0, 0.95)

    return fpar.astype(np.float32)

def calc_ec_lue_gpp(fpar, Emax, VPD0, Par, Ts, VPD):
    Ws_EC = calc_ws_ec(VPD, VPD0)
    if len(Emax.shape) == 2:
        Emax = np.expand_dims(Emax, axis=2)
    GPP = Par * fpar * Emax * np.minimum(Ts, Ws_EC)

    return GPP.astype(np.float32)

def calc_ec_lue_ld_age_gpp(
        dec_data_nor,
        Emax,
        VPD0,
        Par,
        Ts,
        VPD,
        LAIY_nor,
        LAIM_nor,
        LAIO_nor,
        ratio_Y=0.75,
        ratio_M=1.0,
        ratio_O=0.5
):
    LAI_dec = np.clip(
        8.241 * (1.0 - (dec_data_nor )) - 1.991,
        0.0,
        None
    )

    fpar_dec = 0.95 - np.exp(-0.5 * LAI_dec)
    fpar_dec = np.clip(fpar_dec, FPAR_MIN, FPAR_MAX)

    Apar = fpar_dec * Par

    E_age = Emax * (
            ratio_Y * LAIY_nor +
            ratio_M * LAIM_nor +
            ratio_O * LAIO_nor
    )

    Ws_EC = calc_ws_ec(VPD, VPD0)

    GPP = Apar * E_age * np.minimum(Ts, Ws_EC)

    return GPP.astype(np.float32)

ec_lue_gpp_configs = [
    {
        'output_name': 'EC_LUE_EVI',
        'input_type': 'VI',
        'vi_name': 'EVI',
        'input_data': EVI,
        'save_path': r'data/gpp/outputs/EC_LUE_EVI_GPP_local.tif'
    },
    {
        'output_name': 'EC_LUE_kNDVI',
        'input_type': 'VI',
        'vi_name': 'kNDVI',
        'input_data': kNDVI,
        'save_path': r'data/gpp/outputs/EC_LUE_kNDVI_GPP_local.tif'
    },
    {
        'output_name': 'EC_LUE_NDVI',
        'input_type': 'VI',
        'vi_name': 'NDVI',
        'input_data': NDVI,
        'save_path': r'data/gpp/outputs/EC_LUE_NDVI_GPP_local.tif'
    },
    {
        'output_name': 'EC_LUE_MOD_LAI',
        'input_type': 'MOD_LAI',
        'vi_name': None,
        'input_data': MODIS_LAI,
        'save_path': r'data/gpp/outputs/EC_LUE_MODIS_LAI_GPP_local.tif'
    },
    {
        'output_name': 'EC_LUE_Dec_LAI',
        'input_type': 'Dec_LAI',
        'vi_name': None,
        'input_data': dec_data_nor,
        'save_path': r'data/gpp/outputs/EC_LUE_LAI_Dec_GPP.tif'
    },
    {
        'output_name': 'EC_LUE_LD_Age',
        'input_type': 'LD_Age',
        'vi_name': None,
        'input_data': dec_data_nor,
        'save_path': r'data/gpp/outputs/EC_LUE_LAI_Dec_Demography_GPP.tif'
    }
]

print(">>> Generating EC-LUE GPP maps...")

for cfg in ec_lue_gpp_configs:
    output_name = cfg['output_name']
    input_type = cfg['input_type']
    vi_name = cfg['vi_name']

    print("=" * 80)
    print(f"Generating {output_name}")

    if input_type == 'LD_Age':
        print("Using dynamic leaf-age capacity:")
        print(f"  Emax_mature = {Emax_mature:.4f}")
    else:
        print("Using mean leaf-age capacity:")

    print(f"  VPD0 = {VPD0_fixed:.4f}")

    if input_type == 'VI':
        a_vi = VI_FPAR_COEF[vi_name]['a']
        b_vi = VI_FPAR_COEF[vi_name]['b']
        print("Using VI-to-fPAR conversion:")
        print(f"  fPAR = {a_vi:.3f} * {vi_name} + ({b_vi:.3f})")

    if input_type == 'LD_Age':

        GPP = calc_ec_lue_ld_age_gpp(
            dec_data_nor=cfg['input_data'],
            Emax=Emax_mature,
            VPD0=VPD0_fixed,
            Par=Par,
            Ts=Ts,
            VPD=VPD,
            LAIY_nor=LAIY_nor,
            LAIM_nor=LAIM_nor,
            LAIO_nor=LAIO_nor,
            ratio_Y=0.75,
            ratio_M=1.0,
            ratio_O=0.5
        )

    else:

        fpar = build_fpar(
            input_data=cfg['input_data'],
            input_type=input_type,
            vi_name=vi_name
        )

        GPP = calc_ec_lue_gpp(
            fpar=fpar,
            Emax=Emax_mean_leaf,
            VPD0=VPD0_fixed,
            Par=Par,
            Ts=Ts,
            VPD=VPD
        )

    save_tif(GPP, cfg['save_path'], dec_geo, dec_prj, NBANDS_OUT)

    print(f"Saved: {cfg['save_path']}")
    print(f"GPP mean = {np.nanmean(GPP):.3f}")
    print(f"GPP min  = {np.nanmin(GPP):.3f}")
    print(f"GPP max  = {np.nanmax(GPP):.3f}")

print("\nAll EC-LUE GPP maps have been generated using fixed LD-LUE parameters.")
