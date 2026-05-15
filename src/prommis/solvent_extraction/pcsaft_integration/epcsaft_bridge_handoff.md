# ePC-SAFT Bridge Handoff

This integration folder keeps the existing `pcsaft_integration` name, but the bridge it consumes should now be treated as `epcsaft_pse` from `C:\Users\Tanner\Documents\git\pcsaft-pse\src`.

## What The Demo Consumes

- The one-stage PrOMMiS MSContactor POC loads `PCSAFTBlock`, `run_lle_stage`, `SurrogateModel`, `deserialize_surrogate`, and `StageInputs` from `epcsaft_pse`.
- The bridge side provides the Li/Na/Cl surrogate transfer variables and nominal calibration signal.
- PrOMMiS then uses those values to pin the stage partition behavior and expose stream-level outputs through the existing solvent-extraction model.

## Case-Study Framing

- The active benchmark should be the non-ionic HBTA/TOPO produced-water case, not an ionic-liquid story.
- The Gando 2025 selective wrapper is still useful as a surrogate transfer benchmark, but it should be described as a bridge/demo path rather than the final chemistry target.
- Older ionic-liquid-only wording should be treated as historical background and not as the active case-study framing.

## Boundary

- Do not rewrite core IDAES or broad PrOMMiS internals for this integration note.
- Keep this folder as-is unless a future cleanup justifies a wider rename.
