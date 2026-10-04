# Figure workflows

This list separates the input starting point, available outputs, and checks
performed on those outputs. "Executed" means the script completed; it does
not by itself establish numerical agreement or independent validation.
ED denotes Extended Data; SI denotes Supplementary Information.

## Main and Validation Figures

Script paths below are relative to `code/`; input paths are relative to `data/`.

| Figure | Entry script | Input data | Outputs | Verification status |
| --- | --- | --- | --- | --- |
| Main Fig. 1 | None | Conceptual framework | Illustration | Not a computational workflow |
| Main Fig. 2 | `figure2_generate_time_lags.py`; `figure2_plot_seasonality.py`; `reproduce_main_figures.py` | Processed deciduousness, EVI, precipitation, forest mask, site coordinates | Lag raster; seasonal curves; amplitude and lag maps; site tables | Executed. Regenerated lag values and georeferencing match the released reference raster. Site means and uncertainties checked against the reference calculation; amplitude color scale and range checked against the reference plotting code. No claim of pixel-identical manuscript artwork |
| Main Fig. 3 | `figure3_asynchrony_driver_analysis.py`; `figure3_plot_driver_maps.py`; `reproduce_main_text.py` | Processed predictors, deciduousness, time lag, forest mask | Fitted model, SHAP and cross-validation arrays, driver raster, diagnostic plots | Refit executed; driver raster checked against the released reference raster; model summary and driver proportions checked against manuscript targets |
| Main Fig. 4 | `gpp_ec_lue_experiments.py`; `gpp_mod_lue_experiments.py`; `gpp_two_leaf_ec_lue_experiments.py`; `gpp_evaluate_against_sif.py` | Processed meteorology, vegetation indices, leaf-age inputs, calibrated parameters, SIF and site observations | GPP experiment rasters, ensemble comparisons, site statistics | All 12 regenerated GPP rasters match released references; model and ensemble comparison arrays checked against the reference calculation. Improved-pixel proportion is 79.8537%, matching the manuscript's rounded 79.9% |
| ED Fig. 2 | `site_phenocam_evaluation.py` | 250 m composite, existing coarse forest mask, Phenocam tables, BCI satellite table | Five-site curves, comparison panel, metrics CSV and JSON | Final default configuration executed. All five site-wise r and RMSE values equal the supplied reference calculation; mean r = 0.9179836 and mean RMSE = 0.03906885. These are observation-guided spatial-matching statistics |
| ED Fig. 3 | `site_ground_evaluation.py` | Standardized litterfall pairs, site correlations, inventory pairs and GEI tables | Five-panel PNG/PDF, raw inventory fit, confidence bounds, metrics JSON | Executed from processed CSVs. Checked statistics for litterfall, inventory and GEI match the reference calculation. Inventory uses 289 complete records, r = 0.5844362 and residual SD = 0.09216717; fitted-mean confidence bounds also match |
| ED Fig. 7 | `gpp_evaluate_against_sif.py` | Regenerated GPP experiments and processed SIF references | Per-formulation comparisons and distributions | Executed; numerical comparisons checked against the reference calculation |
| SI Fig. 7 | `figure3_asynchrony_driver_analysis.py` | Processed driver-model inputs | Cross-validation figure; `cv_response` and `cv_prediction` in `figure3_shap_outputs.npz` | Exports are implemented and were generated during the driver-model run. Aggregate CV metrics checked; a separate element-by-element comparison of pooled predictions with manuscript source predictions has not been recorded |

## Other Figure-Specific Analyses

"Not included in this release" refers to the complete figure-specific workflow,
not necessarily to every underlying input. Related inputs are identified below;
their sources and access routes are listed in [data provenance](DATA_PROVENANCE.md).
Methods are described in the corresponding manuscript and Supplementary captions.

| Figure | Entry script | Input data | Outputs | Verification status |
| --- | --- | --- | --- | --- |
| ED Fig. 1 | No dedicated entry point | Boundaries and site metadata included | Site overview map | Not included in this release |
| ED Fig. 4 | No dedicated entry point | Some comparison products included | Cross-product comparison | Not included in this release |
| ED Fig. 5 | Related: `figure3_asynchrony_driver_analysis.py` | Eight predictor inputs included | SHAP/model plots | Related outputs executed; exact caption-specific panel not separately checked |
| ED Fig. 6 | Related: driver analysis | Predictors and regional masks included | Regional contributions | Complete figure workflow not included in this release |
| ED Fig. 8 | Related: GPP generators | Monthly GPP products included | Monthly/site panels | Complete figure workflow not included in this release |
| ED Fig. 9 | Related: GPP evaluation | GPP, masks and regional geometry included | Ecoregion attribution | Complete figure workflow not included in this release |
| ED Fig. 10 | Related: GPP experiments | Model products and some site data included | Leaf-age attribution | Complete figure workflow not included in this release |
| SI Fig. 1 | No dedicated entry point | Deciduousness composites included | Threshold-area analysis | Not included in this release |
| SI Fig. 2 | Site unmixing example only | Selected Sentinel-2 scenes and composites included | Resolution/endmember comparison | Full figure workflow not included in this release |
| SI Fig. 3 | No dedicated entry point | Deciduousness and lag included | Binned amplitude-lag analysis | Not included in this release |
| SI Fig. 4 | No dedicated entry point | Regional masks included | Regional histograms | Not included in this release |
| SI Fig. 5 | No dedicated entry point | EVI and GPP kNDVI inputs included | Seasonal index comparison | Full figure workflow not included in this release |
| SI Fig. 6 | No dedicated entry point | Lag and regional inputs included | Peak-to-peak timing | Not included in this release |
| SI Fig. 8 | No dedicated entry point | Core predictors included; MOD11A2 source linked in provenance | LST sensitivity | Not included in this release |
| SI Fig. 9 | Related: driver analysis | Predictors and SHAP dependencies included | Amplitude-stratified fits | Complete figure workflow not included in this release |
| SI Fig. 10 | Related: GPP evaluation | GOSIF and CSIF inputs included | Separate SIF comparisons | Complete figure workflow not included in this release |
| SI Fig. 11 | Related: GPP experiments | GPP products and regional inputs included | Regional annual GPP | Complete figure workflow not included in this release |
| SI Fig. 12 | No dedicated entry point | BRDF EVI included; MAIAC source linked in provenance | EVI product comparison | Not included in this release |
| SI Fig. 13 | No dedicated entry point | BRDF lag inputs included; MAIAC source linked in provenance | Product-specific lag comparisons | Not included in this release |
| SI Fig. 14 | None | Full drone/source-image collection not supplied | Image comparison | Not included in this release |
| SI Fig. 15 | Site example only | Full dynamic-endmember archive not supplied | Basin endmember analysis | Not included in this release |
| SI Fig. 16 | None | K=2 retraining inputs/workflow not supplied | Two-endmember sensitivity | Not included in this release |
| SI Fig. 17 | None | Complete shade-treatment validation inputs not supplied | Shade reassignment sensitivity | Not included in this release |
| SI Fig. 18 | No dedicated entry point | Some Phenocam tables included; full crown imagery/annotations not supplied | Crown-tracking validation | Not included in this release |
| SI Fig. 19 | None | Basin reconstruction-error archive not supplied | Spectral error analysis | Not included in this release |
| SI Fig. 20 | None | Monte Carlo perturbation archive not supplied | Noise sensitivity | Not included in this release |
| SI Fig. 21 | Related: leaf-age and GPP scripts | Leaf-age products and some site inputs included | LAI/site comparison | Complete figure workflow not included in this release |
| SI Fig. 22 | None | Conceptual LD-LUE framework | Illustration | Not a computational workflow |
| SI Fig. 23 | Related: leaf-age model | Some litterfall/leaf-age inputs included | Litterfall-to-SLA and Eq. 13 analysis | Complete figure workflow not included in this release |
| SI Fig. 24 | Related: GPP evaluation | SIF and EC inputs included | Matched-year SIF/EC analysis | Complete figure workflow not included in this release |
| SI Fig. 25 | None | GPP inputs included; perturbation workflow not supplied | Deciduousness uncertainty propagation | Not included in this release |
| SI Fig. 26 | None | Residual-bootstrap workflow not supplied | Eq. 9 uncertainty | Not included in this release |
| SI Fig. 27 | Related: GPP generators | Parameter summaries included; bootstrap workflow not supplied | Site parameter uncertainty | Not included in this release |
| SI Fig. 28 | Related: GPP generators | Baseline inputs included; parameter ensemble not supplied | Basin parameter uncertainty | Not included in this release |
| SI Table 1 | No table assembly entry point | Some site metadata/observations included | Site summary | Complete table workflow not included in this release |
| SI Table 2 | No table assembly entry point | Product sources listed in provenance | Product/access summary | Complete table workflow not included in this release |
| SI Table 3 | No table assembly entry point | Main predictor inputs included | Predictor metadata | Complete table workflow not included in this release |
| SI Table 4 | GPP formulation scripts | Model definitions and parameter summaries included | Formulation summary | Automated table assembly not included in this release |
| SI Table 5 | No calibration entry point | Parameter summary CSVs included | Calibration summary | Calibration and uncertainty workflow not included in this release |

## Site-Level Mapping Example

`sentinel2_site_unmixing.py` was executed on the default Testing input for
2019-2021, producing 87 abundance rasters. GDAL must load before TensorFlow in
the tested Windows environment. Output grids, finite values and abundance sums
were checked. The strict abundance-range check did not pass: some values were
slightly outside [0, 1]. This is an execution test, not an accuracy or
repeat-training reproducibility test; it does not establish basin-wide mapping
reproduction.
