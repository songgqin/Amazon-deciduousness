# Amazon Forest Deciduousness and Phenology

Research code and processed datasets for mapping Amazon forest deciduousness,
characterizing its seasonal relationship with canopy greenness, and evaluating
the role of leaf-age dynamics in gross primary productivity (GPP).

The repository brings together satellite unmixing, site-level validation,
phenological analysis, environmental-driver modeling, and light-use efficiency
(LUE) experiments. It provides executable workflows for selected manuscript
figures and supporting analyses.

## Overview

The analyses address three connected questions:

- How does forest deciduousness vary spatially and seasonally across the Amazon?
- Which environmental predictors are associated with the seasonal asynchrony between deciduousness and vegetation greenness?
- How does representing leaf-age dynamics affect modeled productivity and its agreement with observations?

## Repository Organization

| Location | Contents |
| --- | --- |
| `code/` | Analysis, model, validation, and figure-generation scripts |
| `data/deciduousness/` | Deciduousness composites and seasonal products |
| `data/seasonality/` | EVI and phenological time-lag products |
| `data/drivers/`, `data/climate/` | Environmental predictors and associated spatial inputs |
| `data/gpp/`, `data/leaf_age/` | Productivity inputs, model parameters, leaf-age products, and released model results |
| `data/phenocam/`, `data/validation/` | Site observations and validation tables |
| `data/sentinel2/` | Sentinel-2 inputs for site-level unmixing |
| `data/forest_mask/`, `data/boundaries/` | Land-cover masks and geographic boundaries |
| `data/maintext_release/` | Companion manuscript data release |
| `docs/` | Workflow coverage and data-provenance documentation |
| `outputs/` | Locally generated results; excluded from Git version control |

## Installation

### Obtain the Code and Data

Git LFS is required to retrieve the large raster files. A checkout containing
only LFS pointer files cannot run the analyses.

```bash
git lfs install
git clone https://github.com/songgqin/Amazon-deciduousness.git
cd Amazon-deciduousness
git lfs pull
git lfs fsck
```

The complete LFS checkout is approximately 3.3 GB. The largest tracked input
is `data/deciduousness/Composite_Data_250m_gf_3y.tif` at approximately 2 GB.
Make sure the repository has enough local disk space before running the
250 m validation workflow; a clone containing only LFS pointer files is not a
usable data checkout.

### Create the Python Environment

Use the supplied Conda environment for the tested dependency configuration:

```bash
conda env create -f environment.yml
conda activate amazon-deciduousness
```

The installation has been tested on Windows. Other operating systems may
require platform-specific dependency adjustments. `requirements.txt` lists
dependencies across the codebase; it is not a substitute for the tested
`environment.yml` configuration.

Figure 3 model refitting requires an NVIDIA GPU and the validated CUDA-enabled
XGBoost build. Check the installation before starting the model run:

```bash
python -c "import xgboost as xgb; print('xgboost', xgb.__version__); print(xgb.build_info())"
```

The expected version is `3.0.4`, with a CUDA-enabled build. Then run this
small GPU smoke test:

```bash
python -c "import numpy as np, xgboost as xgb; X=np.array([[0.],[1.]], dtype=np.float32); y=np.array([0,1]); m=xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method='hist', device='cuda'); m.fit(X,y); print('CUDA XGBoost OK')"
```

If the version is not `3.0.4`, `build_info()` does not report CUDA support,
or the smoke test reports that CUDA is unavailable, do not start the Figure 3
refit. Recreate the supplied Conda environment and confirm that the NVIDIA
driver is visible to Python. Saved-product checks and the Figure 3 map-only
workflow do not require refitting the model. Large raster analyses also
require sufficient memory and disk space.

Run the commands below from the repository root. For PyCharm, select the newly
created Conda environment as the project interpreter and use the repository
root as the working directory.

## Scientific Workflows

### Satellite Unmixing

**Purpose:** estimate vegetation fractions from Sentinel-2 imagery to support
deciduousness mapping.

`code/sentinel2_site_unmixing.py` implements the IG-ECAE site-level unmixing
workflow. It uses the supplied Sentinel-2 inputs and associated helper modules.
Outputs include estimated endmember spectra and per-scene abundance rasters.

### Seasonality and Phenological Asynchrony

**Purpose:** characterize the seasonal relationship between deciduousness and
EVI, including spatial variation in deciduousness amplitude and time lag.

`code/figure2_generate_time_lags.py` calculates the time-lag raster.
`code/figure2_plot_seasonality.py` produces the Figure 2 site seasonal curves,
deciduousness-amplitude map, and time-lag map, together with supporting data.

```bash
# Use a new output directory for each command; existing directories are rejected.
python code/figure2_generate_time_lags.py --output-dir outputs/figure2_lag
python code/figure2_plot_seasonality.py --lag-path outputs/figure2_lag/time_lag_map.tif --output-dir outputs/figure2
```

### Environmental Predictors

**Purpose:** assess how environmental conditions relate to phenological
asynchrony and identify spatial patterns in model-predicted contributions.

`code/figure3_asynchrony_driver_analysis.py` fits the XGBoost model and uses
SHAP to characterize predictor contributions. `code/figure3_plot_driver_maps.py`
contains the legacy plotting implementation; the public map/check entry point
is `reproduce_main_text.py`. These analyses describe predictive associations,
not causal effects.

Run the model refit and the manuscript map export as separate steps. The refit
requires the CUDA check above; the map export uses the released or newly
generated driver raster and does not refit XGBoost:

```bash
# Step 1: refit XGBoost and export the driver raster and SHAP/CV products.
# Use a new output directory for every run.
python code/figure3_asynchrony_driver_analysis.py --output-dir outputs/figure3_model_run

# Step 2: plot/check the Figure 3 driver map. Use a new output directory.
python -c "from pathlib import Path; import sys; sys.path.insert(0, 'code'); from reproduce_main_figures import figure3_map; print(figure3_map(strict=True, output_dir=Path('outputs/figure3_map_run'), driver_path=Path('outputs/figure3_model_run/asynchrony_driver_map_3type.tif')))"
```

The refit directory contains `figure3_metrics.json`, `figure3_model.ubj`,
`figure3_shap_outputs.npz`, `asynchrony_driver_map_3type.tif`, and diagnostic
figures. The map export is written as `Main_Figure3e_reproduced.png` with a
companion metrics JSON. The map-only command reads the driver raster produced
in Step 1; `reproduce_main_text.py` remains available when the full released-
input check is desired.

### Leaf-Age Dynamics

**Purpose:** connect deciduousness and leaf area dynamics with the representation
of leaf age in productivity models.

`code/gpp_leaf_age_demography_model.py` estimates leaf-age dynamics from
deciduousness and leaf area index inputs. The resulting leaf-age products provide inputs to the GPP experiments.

### Productivity Experiments

**Purpose:** compare alternative representations of vegetation activity and
leaf-age dynamics across three LUE formulations.

| Script | Function |
| --- | --- |
| `code/gpp_ec_lue_experiments.py` | Generate EC-LUE productivity experiments |
| `code/gpp_mod_lue_experiments.py` | Generate MOD-LUE productivity experiments |
| `code/gpp_two_leaf_ec_lue_experiments.py` | Generate two-leaf EC-LUE productivity experiments |
| `code/gpp_evaluate_against_sif.py` | Evaluate experiments against SIF and eddy-flux benchmarks and generate Figure 4 and Extended Data Figure 7 outputs |

Outputs include productivity rasters, comparison figures, and quantitative
evaluation summaries. Model generation and evaluation can be run together
through the main-figure entry point below.

To evaluate existing GPP products directly against the two SIF products and
eddy-flux observations, use:

```bash
# Use a new output directory for each run.
python code/gpp_evaluate_against_sif.py --gpp-dir data/gpp/outputs --output-dir outputs/gpp_evaluation --strict
```

The evaluator requires nine GPP rasters covering the three formulations, the
two SIF inputs `data/gpp/sif/Amazon_GOSIF_mean_calibrated.tif` and
`data/gpp/sif/CSIF_Amazon_005_calibrated.tif`, and the eddy-flux tables in
`data/validation/eddy_flux/`. The `--strict` option checks that the ensemble
improved-pixel percentage rounds to the manuscript value of 79.9%; without it,
the evaluator still writes the figures, metrics and comparison tables.

### Phenocam Validation

**Purpose:** evaluate satellite-derived deciduousness seasonality against
Phenocam observations at BCI, K34, K67, ATTO, and RJA.

`code/site_phenocam_evaluation.py` uses the released 250 m deciduousness composite
for Amazon sites and the supplied Sentinel-2 table for BCI. It applies the
repository's existing `MCD12Q1_Amazon.tif` forest mask and exports a validation
figure, site-level metrics, and a machine-readable report.

The workflow retains the reference analysis's observation-guided spatial
matching. Its agreement statistics therefore describe that matching procedure,
not an independent held-out spatial validation.

```bash
# The 250 m input is approximately 2 GB; allow several GB of free disk space
# and substantial RAM for raster processing. Use a new output directory.
python code/site_phenocam_evaluation.py --output-dir outputs/phenocam
```

The default input is `data/deciduousness/Composite_Data_250m_gf_3y.tif` and the
default mask is `data/forest_mask/MCD12Q1_Amazon.tif`. Outputs are
`Extended_Data_Fig2_Phenocam_Validation.png`, `phenocam_site_metrics.csv`, and
`phenocam_validation_report.json`.

### Ground-Based Evaluation

**Purpose:** assess satellite-derived deciduousness using field observations
and complementary site-level measurements.

`code/site_ground_evaluation.py` contains the ground-validation analyses and
associated figures. The outputs combine litterfall seasonality, forest-inventory comparisons,
patch-level agreement, and site-level summary distributions.

Run it from the repository root with a new output directory:

```bash
python code/site_ground_evaluation.py --output-dir outputs/ground_validation
```

It reads the processed input tables in `data/validation/ground/`, including
`Litterfall_global_pool_Zscore.csv`, `Litterfall_site_r_values.csv`,
`Plot_Satellite_Dec_Qc_Final.csv`, `GEI_patch_level_data.csv`, and
`GEI_site_level_stats.csv`. It writes
`Extended_Data_Fig3_Combined_Final.png`,
`Extended_Data_Fig3_Combined_Final.pdf`, `inventory_raw_fit.csv`,
`inventory_confidence_band.csv`, and `ground_validation_report.json`.

## Reproducing the Main Results

### Check Released Products

Render and evaluate the supplied main-figure products without refitting the
driver model or regenerating GPP:

```bash
# Use a new output directory for each run.
python code/reproduce_main_text.py --output-dir outputs/saved_product_check
```

### Regenerate from Processed Inputs

Recalculate time lag, refit the Figure 3 model, run the three GPP formulations,
and evaluate the newly computed results:

```bash
# Use a new output directory for each run. This mode refits Figure 3 and
# regenerates the three GPP formulations, so it requires the full environment.
python code/reproduce_main_text.py --regenerate --output-dir outputs/main_figure_run
```

The regeneration workflow compares derived rasters with the released results
and records its checks in `reproduction_report.json`. Figure 2's standalone
three-plot export and Phenocam validation have separate commands above.

## Outputs and Reproduction Scope

Generated figures, rasters, tables, and reports are written locally under
`outputs/`. They are not uploaded to GitHub. Released reference products used
as inputs or comparison targets remain under `data/`.

Every command that writes results requires a new output directory: the scripts
intentionally stop with `FileExistsError` when the requested directory already
exists. Keep generated files separate from the released inputs; do not replace
`data/` products with new runs.

## Scope of This Release

This repository provides code and processed inputs for reproducing the main
analyses and selected validation figures. A site-level Sentinel-2 example is
also included to demonstrate the deciduousness-mapping method. The main
analysis workflows start from the supplied processed data.

Upstream satellite processing, basin-wide unmixing, parameter calibration,
and Supplementary analyses without an entry point in the workflow list are
not included in this release. Figure 1 is a conceptual illustration.
Execution, numerical comparison, and manuscript-statistic checks are reported
separately in the workflow list.

For figure-specific verification and data provenance, see:

- [Figure workflow coverage](docs/FIGURE_WORKFLOW_COVERAGE.md)
- [Data provenance and reuse boundaries](docs/DATA_PROVENANCE.md)
- [Deciduousness input notes](data/deciduousness/README.md)

## License

Original code and associated software documentation are released under the
[MIT License](LICENSE). Retain the copyright and license notice when reusing
or redistributing the software.

The MIT license covers original software and its documentation only.
Author-generated research products have no separate data license designated
in this release; no data-reuse rights are granted through the software license.
Third-party data and adapted code retain their source terms and attribution.
See [data provenance and reuse](docs/DATA_PROVENANCE.md) for product-specific
sources and the permission information still required from the authors.
