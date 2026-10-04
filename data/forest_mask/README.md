# Forest-mask versions

Both files use the public filename `MCD12Q1_Amazon.tif`, but their directories
identify different spatial supports and workflow roles. They must not be
substituted for one another.

| File | Spatial role | Used by | Current default role |
| --- | --- | --- | --- |
| `MCD12Q1_Amazon.tif` | Coarse basin mask aligned with the released 5 km workflows | Figure 2 lag, Figure 3 drivers, Figure 4 evaluation, Phenocam support | Default public mask |
| `native/MCD12Q1_Amazon.tif` | Native-resolution mask aligned with the EC-LUE generation input | Native EC-LUE generation or explicit sensitivity runs | Not the public default |

Class 2 selects evergreen broadleaf forest in these workflows. The public
scripts preserve the resampling rule of each analysis: nearest-neighbour
support for categorical mask transfer, and the original bilinear treatment
where Figure 4 evaluates a continuous field after masking. Changing the mask
or resampling rule changes the valid support and the resulting statistics.

SHA-256 checksums:

- Coarse: `6e3647ca626092857d272c80622a0637d9c9cd0a7edae733dd5c0546d6836b0d`
- Native: `9769f72e85d20573897be405ffbb6bfc782e1c3865e8d46ac66adf0272dc1a76`

The released lag workflow defaults to the coarse mask and matches the
validated manuscript baseline. The native mask remains available only for
workflows that explicitly require its spatial support.

The MIT software licence does not replace MODIS dataset terms. See
[data provenance](../../docs/DATA_PROVENANCE.md) for the product source and
reuse boundary.
