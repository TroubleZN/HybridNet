import os


# Keep the standard sharp variant, but constrain the dominant S4 background term
# and remove repair-rate influence in the HIR head for this dedicated experiment.
os.environ.setdefault("DHCORN_VARIANT", "sharp")
os.environ.setdefault("DHCORN_RUN_NAME", "hir_focus_sharp_bgsmall_norepair_20260412")
# os.environ.setdefault("DHCORN_S4_HIBG_MAX", "0.25")
# os.environ.setdefault("DHCORN_MATERNAL_HEALTH_FIXED", "1.0")
# os.environ.setdefault("DHCORN_MATERNAL_REPAIR_FIXED", "1.0")
# os.environ.setdefault("DHCORN_REPAIR_WEIGHT", "0.0")

import main_hir_focus_sharp  # noqa: F401
