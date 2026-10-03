# Main-text Public Data Release

This directory is an additive copy of the 26 files in the manuscript's public
data directory:

`D:/OneDrive - The University Of Hong Kong/PhD_Projects/Project4_Mapping_Amazon_Basin_Using_GEE/Manuscript/Data_Availablity/Data`

The relative subdirectories and filenames are preserved. GeoTIFF files are
tracked with Git LFS. These files are the manuscript-facing public inputs and
are kept separate from the earlier categorized working data under `data/` so
that existing validated results are not overwritten.

The workstation-specific main-figure source-material bundle is not included.
Use `code/reproduce_main_figures.py` for the repository-relative,
non-interactive raster and visual check. It reads this release directory and
the already generated seasonality/driver inputs under `data/`.
