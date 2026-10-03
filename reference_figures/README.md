# Main-text Figure References

`Main_Figure3_Nature_revision_source.png` is the Figure 3 image extracted
from the Nature-revision figure set used in the Word main text. Its panel e
reports the ternary-map proportions as Light 15.2%, Hydroclimate 51.6%, and
Soil 33.2%.

The checked-in `data/drivers/Asynchrony_shap_map_3type_drivers_0818.tif` is a
byte-identical copy of the available local J-drive 0818 raster, but its
repository-relative calculation gives Light 15.5%, Hydroclimate 46.5%, and
Soil 38.1%. The strict command `python code/reproduce_main_text.py` stops on
this mismatch. This is intentional: the exact revision raster used to create
the Word panel must be supplied before the repository can claim full Figure 3
reproduction.
