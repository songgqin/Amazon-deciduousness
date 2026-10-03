"""Check main-figure products, or regenerate them from released processed inputs.

--regenerate computes lag rasters, refits Figure 3, and runs all three GPP
formulations before plotting/evaluation. It does not rerun satellite unmixing,
parameter calibration or the upstream leaf-age model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from reproduce_main_figures import figure2, figure3_map

def check_raster(actual: Path, expected: Path) -> dict:
    a, b = gdal.Open(str(actual)), gdal.Open(str(expected))
    if a is None or b is None:
        raise FileNotFoundError(f"Cannot compare {actual} and {expected}")
    x, y = a.ReadAsArray(), b.ReadAsArray()
    np.testing.assert_array_equal(x, y, err_msg=f"Raster differs: {actual.name}")
    if a.GetGeoTransform() != b.GetGeoTransform() or a.GetProjection() != b.GetProjection():
        raise ValueError(f"Raster georeferencing differs: {actual.name}")
    return dict(file=actual.name, shape=list(x.shape), values_and_grid_equal=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reproduction")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT / "data" or ROOT / "data" in out.parents:
        raise ValueError("Outputs must not be placed in released data/")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(mode="regenerate" if args.regenerate else "check_saved_products", passed=False,
                  upstream_scope="Released processed rasters, calibrated parameters and leaf-age inputs",
                  commands=[], comparisons=[])

    def command(script, *options):
        cmd = [sys.executable, str(ROOT / "code" / script), *map(str, options)]
        start = time.monotonic()
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        report['commands'].append(dict(script=script, returncode=result.returncode,
                                       elapsed_seconds=time.monotonic() - start))
        result.check_returncode()

    try:
        lag = ROOT / "data/seasonality/time_lag_map.tif"
        drivers = ROOT / "data/drivers/asynchrony_driver_map_3type.tif"
        gpp = ROOT / "data/gpp/outputs"
        if args.regenerate:
            command("figure2_generate_correlation_lags.py", "--output-dir", out / "seasonality")
            lag = out / "seasonality/time_lag_map.tif"
            report['comparisons'].append(check_raster(lag, ROOT / "data/seasonality/time_lag_map.tif"))
            command("figure3_asynchrony_driver_analysis.py", "--output-dir", out / "drivers", "--lag-path", lag)
            drivers = out / "drivers/asynchrony_driver_map_3type.tif"
            report['comparisons'].append(check_raster(drivers, ROOT / "data/drivers/asynchrony_driver_map_3type.tif"))
            gpp = out / "gpp"
            for script in ("gpp_ec_lue_experiments.py", "gpp_mod_lue_experiments.py", "gpp_two_leaf_ec_lue_experiments.py"):
                command(script, "--output-dir", gpp)
            generated = sorted(gpp.glob("*.tif"))
            if len(generated) != 12:
                raise RuntimeError(f"Expected 12 GPP rasters, found {len(generated)}")
            for path in generated:
                report['comparisons'].append(check_raster(path, ROOT / "data/gpp/outputs" / path.name))
        print("Wrote:", figure2(out / "figures", lag_path=lag))
        print("Wrote:", figure3_map(strict=True, output_dir=out / "figures", driver_path=drivers))
        command("gpp_evaluate_against_sif.py", "--gpp-dir", gpp, "--output-dir", out / "figure4", "--strict")
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        (out / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Passed:", report['mode'], "(released processed inputs; not all manuscript/SI workflows)")


if __name__ == "__main__":
    main()
