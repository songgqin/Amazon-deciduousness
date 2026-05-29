# Amazon Deciduousness, Phenology, and GPP Analysis Code

This repository contains cleaned research code for Amazon forest deciduousness mapping, seasonality assessment, driver exploration, leaf-age modeling, Sentinel-2 unmixing, and LUE-based GPP experiments.


## Repository Layout

```text
code/                       Cleaned Python scripts and helper modules
data/                       Categorized local input data
outputs/                    Generated figures and rasters
requirements.txt            Python package requirements
```

## Setup

GDAL, Cartopy, Rasterio, and TensorFlow are easiest to install with Conda because they depend on compiled geospatial and numerical libraries.

```bash
conda create -n amazon-deciduousness python=3.10
conda activate amazon-deciduousness
conda install -c conda-forge gdal cartopy rasterio shapely
pip install -r requirements.txt
```

Run scripts from the repository root so relative `data/...` and `outputs/...` paths resolve correctly. The scripts are divided with `# In[]` markers so they can also be opened and run as cells in PyCharm, Spyder, or Jupyter-style IDE workflows.

## Data Layout

The final `data/` folder is organized by workflow category:

```text
data/boundaries/            Amazon basin boundary shapefile
data/climate/               ERA5, precipitation, PAR, temperature, and VPD inputs
data/deciduousness/         Deciduousness composites and yearly products
data/drivers/               Driver analysis rasters and visualized predictor maps
data/forest_mask/           MODIS land-cover masks and forest percentage data
data/gpp/                   GPP inputs, SIF references, model outputs, and parameters
data/leaf_age/              Leaf-age evaluation arrays and modeled leaf-age products
data/phenocam/              Phenocam and Sentinel-2 site validation tables
data/seasonality/           EVI, correlation, time-lag, and seasonality rasters
data/sentinel2/             Sentinel-2 site/year imagery for unmixing
data/validation/            Eddy-flux and ground-validation tables
```

`data/sentinel2/rja_shared/` keeps a site/year directory structure because `code/sentinel2_site_unmixing.py` iterates over that hierarchy. Duplicate filenames from different source folders were unified by keeping distinct descriptive names:

- `data/seasonality/BRDF_EVI_3years_mean.tif` for Figure 2 seasonality.
- `data/gpp/inputs/GPP_BRDF_EVI.tif` for GPP experiments.
- `data/deciduousness/Composite_Data_5km_gf_3y.tif` for main/GPP workflows.
- `data/deciduousness/Figure2_Composite_Data_5km_gf_3y.tif` for Figure 2 plotting.


## Main Workflows

GPP-related scripts use the shared `gpp_` filename prefix. Savitzky-Golay gap filling is standardized through `code/sg_smooth.py`.

`code/sentinel2_site_unmixing.py` runs the IG-ECAE Sentinel-2 unmixing workflow.

`code/figure2_generate_correlation_lags.py` generates seasonality correlation and time-lag rasters.

`code/figure2_plot_seasonality.py` plots Figure 2 seasonality panels.

`code/figure3_asynchrony_driver_analysis.py` runs the driver analysis for phenological asynchrony.

`code/figure3_plot_driver_maps.py` plots the Figure 3 driver maps.

`code/gpp_leaf_age_demography_model.py` estimates leaf-age demography from deciduousness and LAI inputs.

`code/gpp_ec_lue_experiments.py`, `code/gpp_mod_lue_experiments.py`, and `code/gpp_two_leaf_ec_lue_experiments.py` generate GPP experiments under alternative LUE formulations.

`code/gpp_evaluate_against_sif.py` evaluates GPP experiment outputs against SIF and eddy-flux benchmarks.

`code/site_ground_evaluation.py` and `code/site_phenocam_evaluation.py` run site-level validation analyses.

## Runtime Notes

The workflows are data-intensive and some scripts launch long raster or multiprocessing jobs when run end-to-end. A Miniconda/PyCharm-compatible smoke check is summarized in `docs/runtime_check_miniconda.md`.

Generated files should be written to `outputs/`. Keep large input rasters in `data/` with Git LFS, or store them externally and refill them locally before running the workflows.
