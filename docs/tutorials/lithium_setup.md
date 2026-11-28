# Lithium solvent-extraction tutorial: setup

`lithium_produced_water_solvent_extraction.ipynb` builds lithium solvent-extraction flowsheets with IDAES and PrOMMiS,
closing every stage with ePC-SAFT phase and chemical equilibrium from the
[epcsaft](https://github.com/tannerpolley/epcsaft) Engine. It follows the paper it accompanies:

- a three-stage countercurrent extraction of softened Smackover MS-2 brine, comparing four solvent systems;
- the paper's whole-process comparison on 14 produced waters, including both calibrations of each solvent,
  annual cost accounts and levelized cost with equipment and initial solvent fill;
- an optional appendix with a design optimization and widgets for trying other stage designs.

The models are in `prommis.lithium_extraction`, with the source data, fitted parameters and retained results they read
under its `data/` folder. The notebook checks its stage results and annual-cost reconciliations against those retained
results, which come from the [Lithium_Extraction](https://github.com/tannerpolley/Lithium_Extraction) project,
and stops with an error if one is outside its limit. That project owns the models and data; this copy is written by its
`scripts/export_prommis.py`.

## Environment (64-bit Windows or Linux)

The Engine's conda packages exist for Python 3.13 only, so make a new environment. From a Miniforge prompt, in the
root of this PrOMMiS checkout:

```sh
conda create -n prommis-lithium -c conda-forge -c tannerwpolley python=3.13 epcsaft=1.3.1 reaktoro=2.13.0
conda env config vars set -n prommis-lithium PYTHONUTF8=1
conda activate prommis-lithium
pip install -e . "watertap>=1.8,<1.9" "ipywidgets>=8,<9" jupyterlab
idaes get-extensions
```

The Miniforge installer refuses a folder whose path contains a space; if your Windows user name has one, install
Miniforge to `C:\miniforge3`. Clone the repository with git rather than downloading a zip: PrOMMiS reads its version
from the git history. `PYTHONUTF8=1` makes Python read the retained data files as UTF-8 on Windows. `tannerwpolley` is
an interim channel for the Engine until its conda-forge package is published; after that, `-c conda-forge` alone is
enough.

## Run

```sh
cd docs/tutorials
jupyter lab
```

Open the notebook and choose *Run All*; it takes about 2 minutes. To run it without opening it:

```sh
jupyter nbconvert --to notebook --execute --inplace lithium_produced_water_solvent_extraction.ipynb
```

**As an app.** `voila lithium_produced_water_solvent_extraction.ipynb` (after `pip install voila`) serves the
notebook as a web page that shows only the text, widgets and results, with the code hidden; it runs every cell when the
page opens.

The helper functions the notebook calls are the `lithium_*_tools.py` files beside it. The public epcsaft 1.3.1
conda release was built from Engine commit `1b46e57c`.
