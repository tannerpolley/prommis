from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bridge_src_candidates() -> tuple[Path, Path]:
    repo_root = _repo_root()
    return (
        repo_root.parent / 'pcsaft-pse' / 'src',
        Path(r'C:\Users\Tanner\Documents\git\pcsaft-pse\src'),
    )


def _import_epcsaft_pse_symbols():
    from epcsaft_pse.block import PCSAFTBlock
    from epcsaft_pse.bridge import run_lle_stage
    from epcsaft_pse.surrogate import SurrogateModel, deserialize_surrogate
    from epcsaft_pse.types import StageInputs

    return PCSAFTBlock, run_lle_stage, SurrogateModel, deserialize_surrogate, StageInputs


def _is_missing_epcsaft_pse(exc: ModuleNotFoundError) -> bool:
    return bool(exc.name) and (exc.name == 'epcsaft_pse' or exc.name.startswith('epcsaft_pse.'))


def ensure_epcsaft_pse_imports():
    try:
        return _import_epcsaft_pse_symbols()
    except ModuleNotFoundError as exc:
        if not _is_missing_epcsaft_pse(exc):
            raise

    last_error = None
    for candidate in _bridge_src_candidates():
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            return _import_epcsaft_pse_symbols()
        except ModuleNotFoundError as exc:
            if not _is_missing_epcsaft_pse(exc):
                raise
            last_error = exc

    repo_root = _repo_root()
    sibling_src = repo_root.parent / 'pcsaft-pse' / 'src'
    fallback_src = Path(r'C:\Users\Tanner\Documents\git\pcsaft-pse\src')
    raise RuntimeError(
        'Could not import epcsaft_pse. Tried the installed package, '
        f'sibling checkout at {sibling_src}, and fallback Windows path {fallback_src}. '
        'Install the epcsaft-pse package or place the sibling checkout at one of those locations.'
    ) from last_error
