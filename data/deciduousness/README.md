# Deciduousness inputs

## Revised Figure 2 site curves

`Figure2_Deciduousness_Seasonality.tif` is the three-year mean actually read by
the author's revised Figure 2 site-seasonality calculation. It is copied
byte-for-byte from that calculation's input, with a descriptive public name.
It is not interchangeable with `Composite_Data_5km_gf_3y.tif`, which remains
in use by other workflows. Replacing that shared file would also change the
inputs to unrelated model calculations.

- Shape: 650 rows x 786 columns x 12 bands.
- Band order: January through December, as consumed by the source workflow.
- Storage: UInt16; divide by 1000 to obtain the dimensionless fraction.
- NoData: 65535; the Figure 2 workflow also excludes values above 1000.
- Affine transform: (-79.77497863776252, 0.04491576420595738, 0,
  8.628408135496414, 0, -0.04491576420595738).
- CRS metadata: absent in the original TIFF. The plotting workflow treats
  its coordinates as longitude/latitude. No CRS was invented or written into
  the preserved source file; explicit CRS provenance remains to be confirmed.
- SHA-256: `24e94a892c3b432c062f28cae5da116414ee909c7d1ae5fb679ef226e4f30f4e`.

`Composite_Data_2019_5km_gf.tif`, `Composite_Data_2020_5km_gf.tif` and
`Composite_Data_2021_5km_gf.tif` supply the annual monthly stacks used for
interannual error bars. Each year is averaged over the native 3 x 3 site
neighborhood before computing SD across the three years (`ddof=0`). The mean
curve is read from the supplied three-year mean, not reconstructed by a new
gap-filling or averaging method. All three annual files were verified to be
byte-identical to the revised local Figure 2 inputs.

The companion EVI mean/annual files, precipitation input and lag raster were
also checked against the original local input files by SHA-256. Their public
locations are recorded in `code/reproduce_main_figures.py`.

The repository's MIT software license does not grant rights to these research
datasets. Dataset-specific redistribution and reuse terms remain under audit.
