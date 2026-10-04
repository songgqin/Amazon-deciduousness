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
XGBoost build. Checking the saved main-figure products does not require refitting
that model. Large raster analyses also require sufficient memory and disk space.

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
python code/figure2_generate_time_lags.py --output-dir outputs/figure2_lag
python code/figure2_plot_seasonality.py --lag-path outputs/figure2_lag/time_lag_map.tif --output-dir outputs/figure2
```

### Environmental Predictors

**Purpose:** assess how environmental conditions relate to phenological
asynchrony and identify spatial patterns in model-predicted contributions.

`code/figure3_asynchrony_driver_analysis.py` fits the XGBoost model and uses
SHAP to characterize predictor contributions.
`code/figure3_plot_driver_maps.py` visualizes dominant-predictor patterns for
Figure 3. These analyses describe predictive associations, not causal effects.

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
python code/site_phenocam_evaluation.py --output-dir outputs/phenocam
```

### Ground-Based Evaluation

**Purpose:** assess satellite-derived deciduousness using field observations
and complementary site-level measurements.

`code/site_ground_evaluation.py` contains the ground-validation analyses and
associated figures. The outputs combine litterfall seasonality, forest-inventory comparisons,
patch-level agreement, and site-level summary distributions.

## Reproducing the Main Results

### Check Released Products

Render and evaluate the supplied main-figure products without refitting the
driver model or regenerating GPP:

```bash
python code/reproduce_main_text.py --output-dir outputs/saved_product_check
```

### Regenerate from Processed Inputs

Recalculate time lag, refit the Figure 3 model, run the three GPP formulations,
and evaluate the newly computed results:

```bash
python code/reproduce_main_text.py --regenerate --output-dir outputs/main_figure_run
```

The regeneration workflow compares derived rasters with the released results
and records its checks in `reproduction_report.json`. Figure 2's standalone
three-plot export and Phenocam validation have separate commands above.

## Outputs and Reproduction Scope

Generated figures, rasters, tables, and reports are written locally under
`outputs/`. They are not uploaded to GitHub. Released reference products used
as inputs or comparison targets remain under `data/`.

Use a new output directory for each run to preserve previous results. Keep
generated files separate from the released inputs; do not replace `data/`
products with new runs.

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
