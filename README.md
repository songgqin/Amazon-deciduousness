# Amazon Deciduousness Phenology

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
- `data/deciduousness/Composite_Data_5km_gf_3y.tif` for Figure 2 plotting.
- `data/gpp/inputs/GPP_BRDF_EVI.tif` for GPP experiments.
- `data/deciduousness/Composite_Data_5km_gf_3y.tif` for main/GPP workflows.


## Main Workflows


`code/sentinel2_site_unmixing.py` runs the IG-ECAE Sentinel-2 unmixing workflow.

`code/figure2_generate_correlation_lags.py` generates seasonality correlation and time-lag rasters.

`code/figure2_plot_seasonality.py` plots Figure 2 seasonality panels.

`code/figure3_asynchrony_driver_analysis.py` runs the driver analysis for phenological asynchrony.

`code/figure3_plot_driver_maps.py` plots the Figure 3 driver maps.

`code/gpp_leaf_age_demography_model.py` estimates leaf-age demography from deciduousness and LAI inputs.

`code/gpp_ec_lue_experiments.py`, `code/gpp_mod_lue_experiments.py`, and `code/gpp_two_leaf_ec_lue_experiments.py` generate GPP experiments under alternative LUE formulations.

`code/gpp_evaluate_against_sif.py` evaluates GPP experiment outputs against SIF and eddy-flux benchmarks.

`code/site_ground_evaluation.py` and `code/site_phenocam_evaluation.py` run site-level validation analyses.

## Main-text Reproduction Release

The manuscript-facing source-material bundle is intentionally not included in
this repository. The repository-relative scripts below are the runnable entry
points for the released inputs.

The corresponding public data release is copied, without renaming, under
`data/maintext_release/`. This additive directory contains the 26 files
released with the manuscript and is tracked with Git LFS for raster files.
`data/maintext_release/README.md` records the source and scope.

For a repository-relative, non-interactive raster check and Figure 2/Figure 3e rendering, run:

```bash
python code/reproduce_main_figures.py
```

The generated QA figures are written to `outputs/figures/` and are intentionally ignored by Git.

The strict main-text check is:

```bash
python code/reproduce_main_text.py
```

It uses the local R4_2 Figure 4 calculation in `code/verify_maintext_fig4.py`, including the original 786 x 650 grid, bilinear forest-mask resize, six valid-month threshold, normalized GOSIF-CSIF reference, formulation-balanced EC-LUE/MOD-LUE/TL-EC ensemble, and a hard gate that the original result rounds to 79.9%. It also checks the Figure 3e proportions read from the checked-in ternary driver raster against the Word main-text values. The detailed CSV/JSON QA records are written to `outputs/qa/`, which is ignored by Git.

The existing `code/gpp_evaluate_against_sif.py` remains the broader Figure 4 evaluation and site-validation workflow; it is not used as the strict main-text gate because it also produces additional site-scale outputs.


