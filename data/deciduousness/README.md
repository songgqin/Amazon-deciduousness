# Deciduousness inputs

The files below are processed inputs for different public workflows. They
have related names but are not interchangeable.

## Input roles

| File | Spatial/temporal meaning | Used by | Current default role |
| --- | --- | --- | --- |
| `Figure2_Deciduousness_Seasonality.tif` | Three-year, 12-month seasonal product used for Figure 2 site-seasonality extraction | `figure2_plot_seasonality.py`, `reproduce_main_figures.py` | Figure 2 seasonal curves and amplitude/timing map input |
| `Composite_Data_5km_gf_3y.tif` | Three-year, 12-month 5 km composite used as the shared model input | `figure2_generate_time_lags.py`, driver analysis and GPP workflows | Default 5 km model product |
| `Composite_Data_250m_gf_3y.tif` | Three-year, 12-month 250 m composite for site-level validation | `site_phenocam_evaluation.py` | Default Phenocam input |
| `Composite_Data_2019_5km_gf.tif`, `Composite_Data_2020_5km_gf.tif`, `Composite_Data_2021_5km_gf.tif` | Annual 5 km monthly stacks for interannual spread | `figure2_plot_seasonality.py`, `reproduce_main_figures.py` | Default annual support for Figure 2 uncertainty |
| `data/maintext_release/Deciduousness_Seasonality_Amazon.tif` | Public-release copy of the 5 km three-year product | Main-text release checks and downstream readers | Release copy; do not substitute for the 250 m Phenocam input |

`Figure2_Deciduousness_Seasonality.tif` is preserved byte-for-byte from the
validated Figure 2 input. The three annual files are used to calculate the
monthly interannual spread; the mean curve is read from the supplied
three-year product rather than rebuilt with a new gap-filling method.

The three-year products store 12 monthly bands. The public scripts apply the
scale and NoData handling required by their source calculations. File
checksums are retained below for release verification.

- `Figure2_Deciduousness_Seasonality.tif`: `24e94a892c3b432c062f28cae5da116414ee909c7d1ae5fb679ef226e4f30f4e`
- `Composite_Data_5km_gf_3y.tif`: `b0eb2cbb395eaea4b725f0f30f36731e069e7f77e8fda80a42a8bcf62fb2f4e`
- `Composite_Data_250m_gf_3y.tif`: `5D32D68C7E25ED2D0DCCED587355390D9CC5831E9533B1099559E58665629C2E`

## Phenocam validation

`Composite_Data_250m_gf_3y.tif` is paired with the repository's existing
`data/forest_mask/MCD12Q1_Amazon.tif` mask. The validation script expands that
mask to the 250 m grid with nearest-neighbour resampling, matches the site
observations using the public spatial-search procedure, and writes a figure,
per-site metrics CSV and JSON report.

The final public configuration was checked against the supplied reference
calculation. The five site-wise correlation and RMSE values matched; the
public run reports mean r = 0.9179836 and mean RMSE = 0.03906885. This is the
documented validation configuration for the repository.

The MIT software licence does not grant rights to these research datasets.
Reuse follows the source and author-provided data terms described in
[data provenance](../../docs/DATA_PROVENANCE.md).
