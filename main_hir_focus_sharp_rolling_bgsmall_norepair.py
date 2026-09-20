import os


os.environ.setdefault("DHCORN_RUN_NAME", "hir_focus_sharp_rolling_bgsmall_norepair_20260412")
os.environ.setdefault("DHCORN_S4_HIBG_MAX", "0.2")
os.environ.setdefault("DHCORN_REPAIR_WEIGHT", "0.0")

import main_hir_focus_sharp_rolling  # noqa: F401
