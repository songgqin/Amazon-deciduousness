# Amazon Deciduousness Phenology

This repository contains cleaned research code for Amazon forest deciduousness mapping, seasonality assessment, driver exploration, leaf-age modeling, Sentinel-2 unmixing, and LUE-based GPP experiments.

## License

The authors' original code and associated software documentation are licensed
under the [MIT License](LICENSE). Retain the copyright and license notice when
redistributing the software or substantial portions of it.

This grant does not cover research datasets, manuscript material, or third-party
code and dependencies. Their original licenses and attribution requirements
continue to apply. Dataset-specific redistribution permissions and third-party
provenance are still being audited; inclusion in this repository is not a grant
of MIT rights to those materials.


## Repository Layout

```text
code/                       Cleaned Python scripts and helper modules
data/                       Categorized local input data
outputs/                    Generated figures and rasters
requirements.txt            Python package requirements
```

## Setup

Download the actual raster data with Git LFS, not just the small pointer files:

```bash
git lfs install
git clone https://github.com/songgqin/Amazon-deciduousness.git
cd Amazon-deciduousness
git lfs pull
git lfs fsck
```

The current figure checks were run on Windows using Python 3.13.2, NumPy 2.2.0,
pandas 2.3.1, OpenCV 4.12.0, Matplotlib 3.10.0, Cartopy 0.25.0 and GDAL 3.6.2.
Figure 3 additionally requires the validated XGBoost 3.0.4 / SHAP 0.48.0 setup
and an NVIDIA GPU for the original `gpu_hist` calculation.

`requirements.txt` is an unpinned inventory of dependencies across workflows,
not a tested installation lockfile. A clean-environment installation and full
end-to-end rerun are still pending. The former generic Python 3.10 recipe was
not the environment used for the validated figures and has been removed.

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
- `data/deciduousness/Figure2_Deciduousness_Seasonality.tif` for revised Figure 2 site curves.
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
Use `--output-dir outputs/figures/my_run` to preserve earlier figure outputs.

The Figure 2 site curves use each product's native 3 x 3 pixel neighborhood.
Error bars are the interannual SD (`ddof=0`) of the spatial means for 2019,
2020 and 2021, matching the author's revised local calculation. The plotted
means come from the supplied three-year mean rasters. The generated
`Main_Figure2_site_monthly.csv` records the means, SDs, individual years,
precipitation and lag statistics. The 24 site-month records were compared
against an independent execution of the original local calculation: maximum
absolute differences were 1.5e-8 for deciduousness means, 1.2e-8 for their SDs,
and zero for EVI means/SDs, precipitation and lag statistics. See
[the input notes](data/deciduousness/README.md) for the distinct mean products.

These checks do not establish full publication reproducibility. Figure 1 is a
conceptual illustration. Figure 2 map-generation parity, fresh GPP simulation
for Figure 4, complete Extended Data/Supplementary coverage, dataset licensing,
and clean-environment validation remain under audit. The commands below check
released products; they do not regenerate the entire analysis from raw data.

The strict main-text check is:

```bash
python code/reproduce_main_text.py
```

It uses the local R4_2 Figure 4 calculation in `code/verify_maintext_fig4.py`, including the original 786 x 650 grid, bilinear forest-mask resize, six valid-month threshold, normalized GOSIF-CSIF reference, formulation-balanced EC-LUE/MOD-LUE/TL-EC ensemble, and a hard gate that the original result rounds to 79.9%. It also checks the Figure 3e proportions read from the checked-in ternary driver raster against the Word main-text values. The detailed CSV/JSON QA records are written to `outputs/qa/`, which is ignored by Git.

The existing `code/gpp_evaluate_against_sif.py` remains the broader Figure 4 evaluation and site-validation workflow; it is not used as the strict main-text gate because it also produces additional site-scale outputs.


