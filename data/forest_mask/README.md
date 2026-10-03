# Forest-mask versions

These two inputs are not interchangeable. Both retain the public filename
`MCD12Q1_Amazon.tif`; their directories identify their roles. Class 2 selects
evergreen broadleaf forest in these workflows. Both files carry WGS84
longitude/latitude CRS metadata (EPSG:4326).

| File | Columns x rows | Pixel size (degrees) | Uses |
| --- | --- | --- | --- |
| `MCD12Q1_Amazon.tif` | 785 x 650 | 0.04491576420597607 | Released Figure 2 lag, Figure 3 drivers, Figure 4 evaluation |
| `native/MCD12Q1_Amazon.tif` | 7840 x 6492 | 0.004491576420597608 | Original EC-LUE generation input |

SHA-256:

- Coarse: `6e3647ca626092857d272c80622a0637d9c9cd0a7edae733dd5c0546d6836b0d`.
- Native: `9769f72e85d20573897be405ffbb6bfc782e1c3865e8d46ac66adf0272dc1a76`.

Nearest-neighbor resizing to the source calculation's 786 x 650 grid gives
232,669 coarse-mask forest pixels and 227,732 native-mask forest pixels;
27,881 class decisions differ. The native file was added byte-for-byte from
the local EC-LUE input. It must not replace the coarse mask globally.

The latest local lag-generation script used the native mask, but that support
does not match the final local lag raster used by the manuscript workflow.
The released lag-generator default uses the coarse mask: the regenerated lag
values and all 231,020 valid-pixel locations match the final raster exactly.
An explicit `--forest-mask data/forest_mask/native/MCD12Q1_Amazon.tif` allows
the alternative local-script support to be examined. That is a sensitivity
run, not the manuscript baseline. It changes 27,402 lag-validity locations;
the lag values on common support were identical in the audit.

Resampling and class masking follow each original workflow. In particular,
Figure 4 evaluation uses its original bilinear resize after replacing other
classes by -1; it is not the nearest-neighbor mask used by Figure 3. Changing
these rules changes the support and requires a new scientific comparison.

The MIT software license does not replace MODIS dataset terms. Full dataset
attribution and redistribution documentation remain under audit.
