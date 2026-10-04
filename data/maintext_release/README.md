# Main-text public data release

This directory preserves the relative filenames and subdirectories of the
manuscript-facing processed data package. GeoTIFF files are tracked with Git
LFS. The package is kept separate from the categorized workflow inputs so that
validated products remain available under their documented names.

Use `code/reproduce_main_figures.py` for the repository-relative regeneration
and comparison workflow. It reads this release directory together with the
categorized inputs under `data/` where the figure workflow requires them.

The files are processed research inputs, not a replacement for the original
satellite, field-observation or third-party source archives. Their reuse is
subject to the source and author-provided terms listed in
[data provenance](../../docs/DATA_PROVENANCE.md).
