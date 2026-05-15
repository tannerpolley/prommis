# HBTA/TOPO IDAES Costing Results

This artifact is generated inside the PrOMMiS environment from the Lithium_Extraction HBTA/TOPO costing handoff. It is a formal IDAES/Pyomo flowsheet costing expression layer, not a vendor-grade TEA.

| Scenario | Flow (m3/day) | Li2CO3 (kg/year) | Revenue price ($/kg) | Breakeven ($/kg) | Margin ($/kg) | Net before tax ($/year) |
|---|---:|---:|---:|---:|---:|---:|
| low | 100 | 28968.25 | 12.00 | 21.00 | -9.00 | -260714.30 |
| base | 1000 | 289682.53 | 20.00 | 28.77 | -8.77 | -2539682.69 |
| high | 10000 | 2896825.32 | 35.00 | 44.30 | -9.30 | -26944447.16 |

## Boundary

- Lithium recovery is supplied by the source-regressed Li/Na HBTA/TOPO model and capped to the Shan/Gando 97.17% three-stage anchor before costing.
- Equipment and operating costs are scenario assumptions from the Lithium_Extraction handoff, represented as IDAES/Pyomo expressions here.
- This module is the required downstream costing path; vendor quotes, solvent-loss experiments, and detailed pretreatment sizing remain future TEA inputs.
