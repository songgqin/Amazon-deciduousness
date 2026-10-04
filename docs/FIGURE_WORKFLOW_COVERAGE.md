# Figure workflow coverage

This matrix records what the public repository currently supports for the
caption set in the Nature manuscript and Supplementary Information. It is a
coverage audit, not a claim that every caption has a runnable end-to-end
workflow. A workflow marked **A** has a public entry point and a recorded
validation against the released processed products. **B** means that public
inputs or related code exist, but the complete caption-specific figure has not
passed an independent gate. **C** means that a source workflow, input dataset,
or both still need to be released or rebuilt. **K** denotes a conceptual
illustration rather than a data-generating analysis.

The matrix deliberately excludes the workstation-only `Materials_for_mainfigure`
bundle, original private work-code packages, Word documents, and reference
figure images. A public script reading a released processed raster is not
counted as reconstruction from raw satellite or field observations.

| Caption | Status | Public entry point or available inputs | Remaining boundary |
| --- | --- | --- | --- |
| Main Fig. 1 | K | None; conceptual framework | Artwork is not a computational workflow |
| Main Fig. 2 | B | `code/reproduce_main_figures.py`, `code/figure2_generate_correlation_lags.py`, `code/figure2_plot_seasonality.py`; seasonality, EVI, precipitation, amplitude, lag and site inputs | Validated from released processed inputs; upstream satellite retrieval and the original raw-observation chain are outside this repository |
| Main Fig. 3 | A | `code/figure3_asynchrony_driver_analysis.py`, `code/figure3_plot_driver_maps.py`; public driver inputs and forest mask | Validated model refit and driver raster; plotting helper was repaired in this release |
| Main Fig. 4 | A | `code/reproduce_main_text.py`, three GPP generators and `code/gpp_evaluate_against_sif.py` | Regeneration starts from released processed inputs, calibrated parameters and leaf-age products |
| ED Fig. 1 | B | Public basin boundary, site tables and observation metadata | Dedicated map-generation workflow and complete source-data audit remain |
| ED Fig. 2 | B | `code/site_phenocam_evaluation.py`, phenocam and Sentinel-2 site inputs | Caption-specific figure and paired-observation gate remain |
| ED Fig. 3 | B | `code/site_ground_evaluation.py`, validation tables and public site inputs | Full 47-site, plot, GEI and scene-level reproduction is not yet gated |
| ED Fig. 4 | C | Some driver/reference rasters are public | Cross-product plotting workflow and all product provenance are incomplete |
| ED Fig. 5 | B | Figure 3 XGBoost/SHAP inputs and `code/figure3_asynchrony_driver_analysis.py` | Eight-variable caption-specific plot not independently validated |
| ED Fig. 6 | B | Driver model inputs and regional masks are public | Regional contribution plot needs a dedicated public entry point and gate |
| ED Fig. 7 | A | `code/gpp_evaluate_against_sif.py`; all three formulation comparisons | Validated strict-positive percentages and maps from regenerated GPP |
| ED Fig. 8 | B | Three GPP generators and released 12 monthly GPP products | Caption-specific 12-panel/site plotting workflow remains |
| ED Fig. 9 | B | GPP products, masks and evaluation code | Ecoregion attribution panel and dedicated map workflow remain |
| ED Fig. 10 | B | GPP generators and site observations | Leaf-age attribution plot needs a dedicated public entry point |
| SI Fig. 1 | C | Deciduousness products are public | Threshold-area analysis script and exact source-data provenance remain |
| SI Fig. 2 | C | Public 5-km products only | 10-m Sentinel-2 aggregation and endmember workflow are not public |
| SI Fig. 3 | B | Deciduousness and lag rasters are public | Caption-specific binned analysis and figure gate remain |
| SI Fig. 4 | C | Basin/subregion masks are public | Dedicated regional histogram workflow remains |
| SI Fig. 5 | C | EVI and some seasonal inputs are public | kNDVI product and four-panel generation lineage remain |
| SI Fig. 6 | C | Public lag raster and regional masks | Peak-to-peak calculation workflow remains |
| SI Fig. 7 | B | XGBoost model code and public driver inputs | Exact pooled 10-fold prediction export is not yet public/gated |
| SI Fig. 8 | C | Core driver inputs are public | Daytime LST input and sensitivity workflow remain |
| SI Fig. 9 | B | Driver model inputs and SHAP dependencies are public | Amplitude-stratified model fits and plot remain |
| SI Fig. 10 | B | GPP products, GOSIF/CSIF inputs and evaluation code | Separate GOSIF and CSIF figure workflow remains |
| SI Fig. 11 | B | Twelve GPP products and regional masks are public | Signed regional annual-GPP calculation and plot need a public entry point |
| SI Fig. 12 | C | BRDF EVI is public | MAIAC EVI input and product-comparison workflow remain |
| SI Fig. 13 | C | BRDF lag inputs are public | MAIAC lag workflow and paired regional comparison remain |
| SI Fig. 14 | C | No drone/source-image bundle is released | Drone and co-registered Sentinel-2 source data require provenance/permission review |
| SI Fig. 15 | C | No dynamic endmember archive is released | 76,276-tile endmember data and generation workflow remain |
| SI Fig. 16 | C | No public K=2 retraining workflow/data | Independent two-endmember retraining must be released before reproduction |
| SI Fig. 17 | C | No complete shade-treatment validation workflow | Patch-level redistribution analysis remains |
| SI Fig. 18 | C | Some ATTO phenocam tables are public | Crown tracking images/annotations and figure workflow remain |
| SI Fig. 19 | C | No spectral reconstruction-error archive is released | Tile-level MSE output and generation workflow remain |
| SI Fig. 20 | C | No Monte Carlo archive is released | Noise perturbation inputs/seed/output workflow remain |
| SI Fig. 21 | B | Leaf-age/GPP code and some site observations are public | Site LAI observations and caption-specific plot gate remain |
| SI Fig. 22 | K | None; conceptual LD-LUE flowchart | Artwork is not a computational workflow |
| SI Fig. 23 | C | Litterfall/leaf-age products are partly public | Raw litterfall-to-SLA conversion and Eq. 13 plot workflow remain |
| SI Fig. 24 | B | SIF and EC inputs plus site evaluation code | Caption-specific year-overlap and SIF/EC plot remain |
| SI Fig. 25 | C | LD-LUE/GPP code exists | Monte Carlo deciduousness uncertainty propagation workflow remains |
| SI Fig. 26 | C | Eq. 9 source inputs are not released as a public workflow | 10,000 residual-bootstrap implementation/output remains |
| SI Fig. 27 | C | GPP generators are public | Bootstrap parameters through site GPP workflow remains |
| SI Fig. 28 | C | GPP generators and evaluation inputs are public | Ten parameterizations and basin uncertainty workflow remains |
| SI Table 1 | B | Site metadata and public observation files | Full table assembly and field-data licensing remain |
| SI Table 2 | C | Caption names comparison products | Product metadata/access table and licenses remain |
| SI Table 3 | B | Eight driver rasters and model code are public | LST sensitivity metadata and final table assembly remain |
| SI Table 4 | B | GPP generators and parameter summaries are public | Complete formulation table should be generated from a public metadata script |
| SI Table 5 | B | Parameter summary CSVs are public | MCMC/calibration provenance and uncertainty-generation workflow remain |

## Current acceptance statement

The public repository currently supports validated regeneration or checking of
the main Figure 2, main Figure 3, main Figure 4 and Extended Data Figure 7
workflows from released processed inputs. It does not yet provide complete raw-data
reproduction for all 47 captioned items. The next additions should target the
highest-value **B** rows only when their source inputs and licensing can be
documented; **C** rows must not be represented as reproducible merely because a
similarly named raster or helper script exists.
