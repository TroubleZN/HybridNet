import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Keep the original sharp training script unchanged for the standard run,
# but force this wrapper to use the no-start0 HIR variant.
os.environ["DHCORN_VARIANT"] = "sharp_nostart0"
os.environ.setdefault("DHCORN_RUN_NAME", "hir_focus_sharp_nostart0_20260314")

runpy.run_path(str(ROOT / "main_hir_focus_sharp.py"), run_name="__main__")
