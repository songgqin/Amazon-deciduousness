# Amazon Deciduousness Phenology

Code and processed data for studying Amazon forest deciduousness, phenological
asynchrony, environmental drivers, and the effects of leaf-age dynamics on
gross primary productivity (GPP).

## Repository Structure

- `code/`: analysis, validation, and figure-generation scripts.
- `data/`: processed inputs, observations, model parameters, and released results.
- `docs/`: figure coverage and data-provenance documentation.
- `outputs/`: locally generated figures, rasters, and reports; ignored by Git and not uploaded to GitHub.
- `environment.yml`: tested Windows/Conda environment.

## Installation

Install Git LFS to download the raster data, then create the Conda environment:

```bash
git lfs install
git clone https://github.com/songgqin/Amazon-deciduousness.git
cd Amazon-deciduousness
git lfs pull
conda env create -f environment.yml
conda activate amazon-deciduousness
```

Run commands from the repository root. Figure 3 model refitting requires the
validated CUDA-enabled XGBoost installation and an NVIDIA GPU.
`requirements.txt` is a dependency inventory; use `environment.yml` for installation.

## Analysis Workflows

### Sentinel-2 Unmixing

`code/sentinel2_site_unmixing.py` estimates vegetation fractions from Sentinel-2
site imagery using the IG-ECAE unmixing workflow.

### Seasonality and Time Lag (Figure 2)

`code/figure2_generate_time_lags.py` calculates the seasonal time lag between
deciduousness and EVI. `code/figure2_plot_seasonality.py` produces site seasonal
curves, the deciduousness-amplitude map, and the time-lag map.

```bash
python code/figure2_generate_time_lags.py --output-dir outputs/figure2_lag
python code/figure2_plot_seasonality.py --lag-path outputs/figure2_lag/time_lag_map.tif --output-dir outputs/figure2
```

### Environmental Drivers (Figure 3)

`code/figure3_asynchrony_driver_analysis.py` uses XGBoost and SHAP to assess
environmental predictors of phenological asynchrony.
`code/figure3_plot_driver_maps.py` visualizes the spatial distribution of
dominant predictors.

### Leaf-Age Modeling

`code/gpp_leaf_age_demography_model.py` estimates leaf-age dynamics from
deciduousness and leaf area index for use in productivity modeling.

### GPP Experiments and Evaluation (Figure 4)

The following scripts generate GPP experiments with alternative light-use
efficiency formulations:

- `code/gpp_ec_lue_experiments.py`: EC-LUE experiments.
- `code/gpp_mod_lue_experiments.py`: MOD-LUE experiments.
- `code/gpp_two_leaf_ec_lue_experiments.py`: two-leaf EC-LUE experiments.

`code/gpp_evaluate_against_sif.py` compares the GPP experiments against SIF and
eddy-flux benchmarks and generates Figure 4 and Extended Data Figure 7 outputs.

### Phenocam Validation

`code/site_phenocam_evaluation.py` compares satellite-derived deciduousness
seasonality with Phenocam observations. It uses the released 250 m composite
and `data/forest_mask/MCD12Q1_Amazon.tif`, and exports a validation figure,
site-level metrics, and a report.

```bash
python code/site_phenocam_evaluation.py --output-dir outputs/phenocam
```

### Ground Validation

`code/site_ground_evaluation.py` evaluates satellite-derived deciduousness
against field observations and site-level validation data.

## Main-Figure Reproduction

Check and render released products without refitting models:

```bash
python code/reproduce_main_text.py --output-dir outputs/saved_product_check
```

Regenerate the time-lag map, refit the Figure 3 model, run the GPP experiments,
and evaluate the newly generated results:

```bash
python code/reproduce_main_text.py --regenerate --output-dir outputs/main_figure_run
```

Choose a new output directory for each run to preserve previous results.
These workflows start from released processed inputs, not raw satellite and
field observations. Complete Extended Data and Supplementary reproduction,
upstream processing, and data permissions remain under review. See the
[figure coverage summary](docs/FIGURE_WORKFLOW_COVERAGE.md) and
[data provenance notes](docs/DATA_PROVENANCE.md) for the current scope.

## License and Citation

Original code and software documentation are available under the
[MIT License](LICENSE). This license does not cover research datasets or
third-party code; their respective permissions and attribution requirements
apply. Data reuse terms remain under review.

Citation metadata are provided in [CITATION.cff](CITATION.cff).


