"""Data root of the lithium-extraction package and the Engine's library file name.

The models read retained inputs as ``REPO_ROOT / "<path>"``, where the path is relative to the Lithium_Extraction
project that owns these models and data. The data root keeps those relative paths, so retained records and
fingerprinted inputs stay byte-identical to that project's copies.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent / "data"
# The Engine's AMPL external-function library, next to its Python package.
ENGINE_ASL_LIBRARY = "epcsaft_asl.dll" if os.name == "nt" else "libepcsaft_asl.so"
ENGINE_CONDA_RELEASES = {"1.3.1": "1b46e57ce8f78b79891b5fe5293917c58fa0f671"}
