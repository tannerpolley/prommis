# Gando 2025 Three-Stage Crossflow in PrOMMiS

## Basis

- Feed: Li = 60.000 mg/L, Na = 10900.000 mg/L
- Organic loading: TOP = 0.015000 mol/L, O/A = 1.000
- Process structure: three sequential crossflow stages with fresh organic each stage
- Chemistry bridge: ePC-SAFT bridge variables and a constant surrogate map into one-stage PrOMMiS solvent extraction solves
- Case-study path: non-ionic HBTA/TOPO produced-water extraction should be the active framing; this Gando wrapper remains a surrogate benchmark, not the final chemistry story

## Cumulative Extraction Comparison

| Stage | Paper Li cumulative (%) | Calc Li stage (%) | Calc Li cumulative (%) | Calc Na stage (%) | Calc Na cumulative (%) | D_Li | Li/Na selectivity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.9500 | 52.0047 | 52.0047 | 0.9067 | 0.9067 | 1.0835 | 118.4138 |
| 2 | 85.6000 | 68.4342 | 84.8499 | 1.2021 | 2.0979 | 2.1680 | 178.1839 |
| 3 | 97.1700 | 85.4948 | 97.8025 | 1.5166 | 3.5827 | 5.8941 | 382.7454 |

## Interpretation

- The PrOMMiS-integrated Li cumulative profile follows the published Gando 2025 trend closely across all three stages.
- Sodium extraction remains low, which preserves the lithium-selective character of the showcase.
- The key integration idea is stagewise: the ePC-SAFT/surrogate transfer variables set the Li target, and each PrOMMiS stage converts that target into stream-level process outputs.
- The active case-study direction is non-ionic HBTA/TOPO oil-and-gas-field-water extraction; ionic-liquid-only framing should be treated as historical background.
