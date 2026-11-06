# RJA Sentinel-2 example

The year folders contain 48 processed multiband Sentinel-2 scenes from the
earlier RJA site analysis: 13 in 2019, 21 in 2020, and 14 in 2021. The
files preserve the source pixel values and spatial grids. The unmixing code
uses GeoTIFF bands 1, 2, 3, and 7 (blue, green, red, and near-infrared).
See [data provenance](../../../docs/DATA_PROVENANCE.md) for the source and
reuse terms.

`scene_inventory.csv` lists the scene dates and SHA-256 checksums. All 48
scenes are used for the annual training pools, subject to the pixel-level
EVI filter. The `legacy_comparison` flag identifies the 47 dates with
abundance rasters in the earlier RJA comparison series. The additional scene
of 2021-10-02 is included in the all-scenes summary.

`reference_endmembers.csv` contains the annual matrices used in the checked
example. Its rows are the four input bands in order and its three abundance
columns are shade, deciduous and evergreen. The checked training run used
the initialization fallback in 2019 and 2020, as no epoch passed the
spectral ordering check; it used the lowest-error eligible epoch in 2021.
The default command refits the endmembers from the supplied scenes with
TensorFlow initialization seed 42 on a single CPU thread. The seed is logged
in `training_summary.csv`. An alternative `--initialization-seed` changes the
network initialization but not the training pixels or their order. The
default refit can differ from the reference replay. Use
`--use-reference-endmembers` to replay the checked abundance calculation.

The EVI filter applies to endmember training only. For the seasonality
summary, each solved pixel is assigned to the largest of its three
abundances after bounding values to [0, 1] for classification. Shade-dominant
pixels are excluded from the denominator. The scene fraction is the count of
deciduous-dominant pixels divided by the sum of deciduous- and
evergreen-dominant pixels. The monthly curve averages scenes within each
year and month, then gives each available year equal weight. Months with
no observations are left blank.
