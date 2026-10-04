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
requirements.txt            Unpinned package inventory
environment.yml             Tested Windows/Conda installation recipe
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

The tested Windows installation uses `environment.yml` (Python 3.13.2,
NumPy 2.2.6, pandas 2.3.1, OpenCV 4.10.0, Matplotlib 3.10.0, Cartopy 0.25.0,
GDAL 3.6.2, XGBoost 3.0.4, SHAP 0.48.0 and TensorFlow 2.20.0). Create it in a
new environment rather than cloning an existing environment:

```bash
conda env create -f environment.yml
conda activate amazon-deciduousness
```

Figure 3 additionally requires the validated XGBoost 3.0.4 CUDA build and an
NVIDIA GPU for the original `gpu_hist` calculation. The saved-product checks
and the other workflows do not require a GPU.

`requirements.txt` remains an unpinned inventory of dependencies across
workflows. `environment.yml` is the tested installation recipe for the public
code; platform-specific solver builds can still vary. The former generic
Python 3.10 recipe was not the environment used for the validated figures and
has been removed.

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
data/seasonality/           EVI, time-lag, and seasonality rasters
data/sentinel2/             Sentinel-2 site/year imagery for unmixing
data/validation/            Eddy-flux and ground-validation tables
```

`data/sentinel2/rja_shared/` keeps a site/year directory structure because `code/sentinel2_site_unmixing.py` iterates over that hierarchy. Duplicate filenames from different source folders were unified by keeping distinct descriptive names:

- `data/seasonality/BRDF_EVI_3years_mean.tif` for Figure 2 seasonality.
- `data/deciduousness/Figure2_Deciduousness_Seasonality.tif` for revised Figure 2 site curves.
- `data/deciduousness/Composite_Data_250m_gf_3y.tif` for 250 m Phenocam validation.
- `data/gpp/inputs/GPP_BRDF_EVI.tif` for GPP experiments.
- `data/deciduousness/Composite_Data_5km_gf_3y.tif` for main/GPP workflows.


## Main Workflows


`code/sentinel2_site_unmixing.py` runs the IG-ECAE Sentinel-2 unmixing workflow.

`code/figure2_generate_time_lags.py` generates only the Figure 2 time-lag
raster. The cyclic correlation is an internal lag-selection calculation; no
correlation map, p-value map, rainfall correlation, or PAR correlation output is
written.

`code/figure2_plot_seasonality.py` generates the Figure 2 seasonality curves,
deciduousness-amplitude map, and categorical time-lag map from repository data.
Use `--error-bars spatial` to reproduce the legacy 3 x 3 spatial-SD variant;
the default `--error-bars interannual` is the manuscript-facing revised result.
For a complete Figure 2 regeneration in a new output directory:

```bash
python code/figure2_generate_time_lags.py --output-dir outputs/figure2_lag
python code/figure2_plot_seasonality.py --lag-path outputs/figure2_lag/time_lag_map.tif --output-dir outputs/figure2
```

`code/figure3_asynchrony_driver_analysis.py` runs the driver analysis for phenological asynchrony.

`code/figure3_plot_driver_maps.py` plots the Figure 3 driver maps.

`code/gpp_leaf_age_demography_model.py` estimates leaf-age demography from deciduousness and LAI inputs.

`code/gpp_ec_lue_experiments.py`, `code/gpp_mod_lue_experiments.py`, and `code/gpp_two_leaf_ec_lue_experiments.py` generate GPP experiments under alternative LUE formulations.

These generators write to `outputs/gpp/` by default, never to released
`data/gpp/outputs/`. Use `--output-dir` for a new run. Existing result files
are not overwritten. The EC-LUE generator uses the original native forest
mask; the Figure 3 and Figure 4 evaluation masks remain unchanged. See
[forest-mask provenance](data/forest_mask/README.md).

`code/gpp_evaluate_against_sif.py` evaluates GPP experiment outputs against SIF and eddy-flux benchmarks.

`code/site_ground_evaluation.py` and `code/site_phenocam_evaluation.py` run site-level validation analyses. The Phenocam workflow defaults to the 250 m composite and the existing `data/forest_mask/MCD12Q1_Amazon.tif` mask; it writes the validation figure, site metrics CSV, and JSON report to a new `outputs/phenocam/` subdirectory.

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
conceptual illustration. Complete Extended Data/Supplementary coverage,
upstream calibration/unmixing, dataset licensing and clean-environment
validation remain under audit. None of the commands below regenerates the
entire analysis from raw satellite and field observations.

To render and check the released products without refitting models:

```bash
python code/reproduce_main_text.py --output-dir outputs/saved_product_check
```

To regenerate the lag map, refit Figure 3, generate all 12 GPP experiment
rasters, and then render/evaluate the newly computed results:

```bash
python code/reproduce_main_text.py --regenerate --output-dir outputs/main_figure_run
```

Choose a new output directory for each run. `--regenerate` requires the
validated XGBoost/GPU environment. It starts from the supplied processed
rasters, calibrated parameter tables and leaf-age inputs; it does not rerun
their upstream calibration or satellite unmixing. Raster values, missing-value
support and georeferencing are compared against released results. Failures
return a nonzero exit status and are recorded in `reproduction_report.json`.

Figure 4 uses the original 786 x 650 grid, bilinear forest-mask resize,
six-valid-month thresholds, normalized GOSIF-CSIF reference and balanced
EC-LUE/MOD-LUE/TL-EC ensemble. The acceptance check requires the strict-positive
pixel proportion to round to 79.9%. Histograms replace obsolete pie insets;
zero changes are not counted as improvements. Site uncertainty is SD with
`ddof=1` across three formulations divided by sqrt(3), not across the six
individual conventional inputs. Per-formulation monthly values, means, SEM,
r/RMSE and basin distribution metrics are exported beside the figures.

Independent validation of the regenerated GPP found all 12 arrays and their
georeferencing identical to the released products. Re-executing the original
local Figure 4 calculation on its original inputs gave identical three-model
and ensemble Delta-r arrays. The 48 site-month observations, means and SEM
matched within CSV floating-point round-trip precision. The ensemble result
was 175,104 improved pixels out of 219,281 (79.8537%, reported as 79.9%).

For Figure 4 alone, selecting a newly generated GPP directory explicitly:

```bash
python code/gpp_evaluate_against_sif.py --gpp-dir outputs/gpp --output-dir outputs/figure4_run --strict
```

`code/verify_maintext_fig4.py` remains an independent check of the saved
Figure 4 products. Its output alone is not evidence of GPP regeneration.


