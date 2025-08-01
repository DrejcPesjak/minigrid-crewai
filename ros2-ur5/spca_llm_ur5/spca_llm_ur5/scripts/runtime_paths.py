# spca_llm_ur5/scripts/runtime_paths.py
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# read-only resources shipped with the package
PKG_ROOT  = Path(__file__).resolve().parent.parent           # …/spca_llm_ur5
SEED_ACTS = PKG_ROOT / "actions" / "agent_actions.py"

RESET_SCRIPT = PKG_ROOT / "scripts" / "reset_env.sh"

PKG_SHARE = Path(get_package_share_directory('spca_llm_ur5'))

# write-able per-user cache
RUN_DIR   = Path.home() / ".spca_llm_ur5"
RUN_DIR.mkdir(exist_ok=True)

# runtime copies
BASE_ACTS = RUN_DIR / "agent_actions.py"  # working copy
TMP_ACTS  = RUN_DIR / "agent_actions_tmp.py" # temporary merge base

# first launch --copy seed if nothing there yet
if not BASE_ACTS.exists():
    BASE_ACTS.write_text(SEED_ACTS.read_text())
# temporary merge base
if not TMP_ACTS.exists():
    TMP_ACTS.write_text(SEED_ACTS.read_text())