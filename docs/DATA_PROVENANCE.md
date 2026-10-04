# Data provenance and redistribution boundary

The MIT license in the repository applies to original code and documentation
only. It does not grant rights to research datasets, satellite products, field
observations, third-party software, or manuscript material. Users must follow
the provider terms for every input before redistributing a copy or depositing a
derived archive.

| Repository group | Scientific role | Current provenance status |
| --- | --- | --- |
| `data/deciduousness/` | Sentinel-2-derived deciduousness composites | Derived research products; byte-level lineage to the author's processed inputs is recorded, but source-observation redistribution terms are still under audit |
| `data/seasonality/` | MODIS EVI, annual EVI and lag products | Processed products and derived lag output; provider attribution and derivative-data terms must be retained |
| `data/climate/`, `data/drivers/inputs/` | ERA5/ERA5-Land and environmental predictors | Mixed third-party products; each product requires its own provider citation and terms check |
| `data/forest_mask/` | MODIS land-cover forest masks | MODIS-derived; the coarse and native files have distinct workflow roles, documented in `data/forest_mask/README.md` |
| `data/gpp/` | Model inputs, calibrated parameters, SIF references and released outputs | Mixed derived and third-party data; the MIT code license does not cover these files |
| `data/leaf_age/` | Leaf-demography and litterfall-derived inputs | Derived research product; raw litterfall and SLA sources still require separate attribution |
| `data/phenocam/`, `data/validation/`, `data/maintext_release/Ground_Observations/` | Phenocam, eddy-flux and ground validation observations | Site-level observations; permission, citation and any restrictions must be checked before reuse |
| `data/sentinel2/` | Site/year Sentinel-2 imagery for unmixing | Third-party remote-sensing inputs; upstream licensing and complete source workflow are not yet part of this release |
| `data/boundaries/` | Basin and regional boundary geometry | Boundary source and redistribution terms should be cited with any map reuse |

## Reproducibility scope

The validated main-figure commands begin from the processed rasters, calibrated
parameter tables and leaf-age products included in this repository. They do not
rerun satellite calibration, dynamic endmember extraction, field-image
annotation, upstream Sentinel-2 unmixing, or raw observation ingestion. The
public release therefore supports processed-input regeneration for the covered
workflows, not full raw-observation reproduction.

Before a formal archive or DOI release, complete these checks:

1. Record the authoritative provider/product citation and license or terms URL
   for every third-party file group.
2. Confirm that field observations, high-resolution imagery and derived files
   may be redistributed in the selected repository.
3. Add per-file or per-group checksums and acquisition/version dates without
   exposing workstation paths.
4. Preserve the distinction between original code under MIT and data under
   provider-specific terms.

No `Materials_for_mainfigure` files, Word documents, reference figures, or
private source-code bundles are part of this repository.
