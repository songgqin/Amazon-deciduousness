"""MOD-LUE GPP experiment runner.

"""

# In[] Imports
import argparse
import sys
from pathlib import Path
import copy
import re
import os
import numpy as np
import cv2
import pandas as pd
from osgeo import gdal
import matplotlib

# In[] Workflow
ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gpp")
args = parser.parse_args()
OUTPUT_DIR = args.output_dir.resolve()
if OUTPUT_DIR == ROOT / "data" or ROOT / "data" in OUTPUT_DIR.parents:
    raise ValueError("Generated outputs must not overwrite released data/ inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    if Path(savePath).exists():
        raise FileExistsError(f"Output exists; choose a new --output-dir: {savePath}")

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

raws_y, columns_x = 786, 650

T_path = r'data/climate/Temp2019_2021_ERA.tif'
Par_path = r'data/climate/ERA5_PAR_total_MJ_2019_2021_Avg_12bands.tif'

VPD_path = r'data/climate/VPD_Amazon.tif'
LAI_path = r'data/gpp/inputs/Amazon_MOD15_LAI.tif'
leaf_age_path = r'data/leaf_age/Leaf_Age_ln_Dec_Litterfall_LAI.tif'

dec_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'

param_summary_path = r'data/gpp/parameters/MOD_LUE_Models_Parameters_Opt_Summary.csv'

out_mod_lai_path = str(OUTPUT_DIR / 'MOD_LUE_GPP_local.tif')
out_dec_lai_path = str(OUTPUT_DIR / 'MOD_LUE_LAI_Dec_GPP.tif')
out_ld_age_path = str(OUTPUT_DIR / 'MOD_LUE_LAI_Dec_Demography_GPP.tif')

dec_geo, dec_prj, dec_data = readTif_gdal(dec_path)
dec_data = resize_to_target(dec_data, raws_y, columns_x)
dec_data = np.array(dec_data, dtype=np.float32)
dec_data[dec_data > 1000] = np.nan
dec_data_nor = dec_data / 1000.0

_, _, T_k = readTif_gdal(T_path)
_, _, VPD = readTif_gdal(VPD_path)
_, _, LAI = readTif_gdal(LAI_path)
_, _, Par = readTif_gdal(Par_path)
_, _, LA = readTif_gdal(leaf_age_path)

T_k = resize_to_target(T_k, raws_y, columns_x).astype(np.float32)
T = T_k - 273.15

VPD = VPD * 0.1
VPD = resize_to_target(VPD, raws_y, columns_x).astype(np.float32)
LAI = LAI.astype(np.float32)
LAI = 0.1 * LAI

LAI = resize_to_target(LAI, raws_y, columns_x).astype(np.float32)
LAI[LAI < 0] = np.nan

Par = resize_to_target(Par, raws_y, columns_x).astype(np.float32)

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

ratio_Y, ratio_M, ratio_O = 0.75, 1.0, 0.50

age_factor = (
    ratio_Y * LAIY_nor +
    ratio_M * LAIM_nor +
    ratio_O * LAIO_nor
)

mean_age_factor = np.nanmean(age_factor,axis=2)

param_df = pd.read_csv(param_summary_path, encoding='utf-8-sig')

def parse_param_from_text(param_text, param_name):
    pattern = rf'{param_name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    match = re.search(pattern, str(param_text))

    if match is None:
        raise ValueError(f"Cannot find {param_name} in parameter text: {param_text}")

    return float(match.group(1))

def get_mod_lue_params(param_df, model_name):
    row = param_df.loc[param_df['Model'] == model_name]

    if len(row) == 0:
        raise ValueError(f"Model {model_name} not found in parameter CSV.")

    row = row.iloc[0]

    if all(col in param_df.columns for col in ['Emax_median', 'VPDmax_median', 'VPDmin_median']):
        Emax = float(row['Emax_median'])
        VPDmax = float(row['VPDmax_median'])
        VPDmin = float(row['VPDmin_median'])
    else:
        param_text = row['Optimized Parameters (Median with 95% CI)']
        Emax = parse_param_from_text(param_text, 'Emax')
        VPDmax = parse_param_from_text(param_text, 'VPDmax')
        VPDmin = parse_param_from_text(param_text, 'VPDmin')

    return Emax, VPDmax, VPDmin

def calc_mod_lue_Ts(T):
    Tmax = 9.09
    Tmin = -8.0

    Ts = np.where(
        T <= Tmin,
        0.0,
        np.where(
            T >= Tmax,
            1.0,
            (T - Tmin) / (Tmax - Tmin)
        )
    )

    return Ts.astype(np.float32)

def calc_mod_lue_Ws(VPD, VPDmax, VPDmin):
    if VPDmax <= VPDmin:
        raise ValueError(f"Invalid VPD parameters: VPDmax={VPDmax}, VPDmin={VPDmin}")

    Ws = np.where(
        VPD <= VPDmin,
        1.0,
        np.where(
            VPD >= VPDmax,
            0.0,
            (VPDmax - VPD) / (VPDmax - VPDmin)
        )
    )

    return Ws.astype(np.float32)

def build_mod_lue_apar(input_type, LAI, dec_data_nor, Par):
    if input_type == 'MOD_LAI':
        fpar = 0.95 - np.exp(-0.5 * LAI)

    elif input_type in ['Dec_LAI', 'LD_Age']:
        LAI_est = np.clip(8.241 * (1.0 - dec_data_nor) - 1.991, 0.0, None)
        fpar = 0.95 - np.exp(-0.5 * LAI_est)

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    fpar = np.clip(fpar, 0.0, 0.95)

    Apar = fpar * Par

    return Apar.astype(np.float32)

def calc_mod_lue_gpp(Emax, VPDmax, VPDmin, Apar, Ts, VPD):

    if len(Emax.shape)==2:
        Emax = Emax[:, :, np.newaxis]

    Ws = calc_mod_lue_Ws(VPD, VPDmax, VPDmin)
    GPP = Apar * Emax * Ts * Ws
    return GPP.astype(np.float32)

def calc_mod_lue_ld_age_gpp(
    Emax,
    VPDmax,
    VPDmin,
    Apar,
    Ts,
    VPD,
    LAIY_nor,
    LAIM_nor,
    LAIO_nor,
    ratio_Y=0.75,
    ratio_M=1.0,
    ratio_O=0.50
):
    Ws = calc_mod_lue_Ws(VPD, VPDmax, VPDmin)

    E_age = (
        Emax * ratio_Y * LAIY_nor +
        Emax * ratio_M * LAIM_nor +
        Emax * ratio_O * LAIO_nor
    )

    GPP = Apar * E_age * Ts * Ws

    return GPP.astype(np.float32)

Ts = calc_mod_lue_Ts(T)

PARAM_SOURCE_MODEL = 'MOD_LUE_LD_Age'

Emax_mature, VPDmax_fixed, VPDmin_fixed = get_mod_lue_params(
    param_df,
    PARAM_SOURCE_MODEL
)

Emax_mean_leaf = Emax_mature * mean_age_factor

print("=" * 80)
print("Using fixed MOD-LUE LD-framework parameters for all experiments")
print(f"Parameter source : {PARAM_SOURCE_MODEL}")
print(f"Emax_mature      = {Emax_mature:.4f}")

print(f"VPDmax_fixed     = {VPDmax_fixed:.4f}")
print(f"VPDmin_fixed     = {VPDmin_fixed:.4f}")
print("=" * 80)

gpp_configs = [
    {
        'output_name': 'MOD_LUE_MOD_LAI',
        'input_type': 'MOD_LAI',
        'save_path': out_mod_lai_path
    },
    {
        'output_name': 'MOD_LUE_Dec_LAI',
        'input_type': 'Dec_LAI',
        'save_path': out_dec_lai_path
    },
    {
        'output_name': 'MOD_LUE_LD_Age',
        'input_type': 'LD_Age',
        'save_path': out_ld_age_path
    }
]

for cfg in gpp_configs:
    output_name = cfg['output_name']
    input_type = cfg['input_type']

    print("=" * 80)
    print(f"Generating {output_name}")

    Apar = build_mod_lue_apar(
        input_type=input_type,
        LAI=LAI,
        dec_data_nor=dec_data_nor,
        Par=Par
    )

    if input_type == 'LD_Age':

        print("Using dynamic leaf-age capacity")
        print(f"Emax_mature = {Emax_mature:.4f}")
        print(f"VPDmax      = {VPDmax_fixed:.4f}")
        print(f"VPDmin      = {VPDmin_fixed:.4f}")

        GPP = calc_mod_lue_ld_age_gpp(
            Emax=Emax_mature,
            VPDmax=VPDmax_fixed,
            VPDmin=VPDmin_fixed,
            Apar=Apar,
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

        print("Using mean leaf-age capacity")

        print(f"VPDmax         = {VPDmax_fixed:.4f}")
        print(f"VPDmin         = {VPDmin_fixed:.4f}")

        GPP = calc_mod_lue_gpp(
            Emax=Emax_mean_leaf,
            VPDmax=VPDmax_fixed,
            VPDmin=VPDmin_fixed,
            Apar=Apar,
            Ts=Ts,
            VPD=VPD
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

print("\nAll MOD-LUE GPP maps have been generated using fixed LD-framework parameters.")
