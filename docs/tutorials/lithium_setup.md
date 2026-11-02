# Lithium solvent-extraction tutorials: setup

Two notebooks build lithium solvent-extraction flowsheets with IDAES and PrOMMiS, closing every stage with ePC-SAFT
phase and chemical equilibrium from the [epcsaft](https://github.com/tannerpolley/epcsaft) Engine:

- `lithium_solvent_comparison.ipynb`: a three-stage countercurrent extraction of softened Smackover MS-2 brine,
  comparing four solvent systems, with the Engine's derivatives shown inside the model and a design optimization.
- `lithium_produced_water_process.ipynb`: a whole process from raw produced water to lithium carbonate (softening,
  extraction, scrub, strip, acid recovery, carbonation and operating cost) for two waters.

The models are in `prommis.lithium_extraction`, with the source data, fitted parameters and retained results they read
under its `data/` folder. Each notebook checks every number it reproduces against those retained results, which come
from the [Lithium_Extraction](https://github.com/tannerpolley/Lithium_Extraction) project, and stops with an error if
one is outside its limit. That project owns the models and data; this copy is written by its
`scripts/export_prommis.py`.

## Environment (64-bit Windows or Linux)

The Engine's conda packages exist for Python 3.13 only, so make a new environment. From a Miniforge prompt, in the
root of this PrOMMiS checkout:

```sh
conda create -n prommis-lithium -c conda-forge -c tannerwpolley python=3.13 epcsaft=1.2.0 reaktoro=2.13.0
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

The produced-water notebook softens each water with the geochemistry library Reaktoro, run as a separate process. It
uses the notebook's own Python, which has Reaktoro in this environment; to use another, set `LITHIUM_SOFTENING_PYTHON`
to it before starting Jupyter.

## Run

```sh
cd docs/tutorials
jupyter lab
```

Open a notebook and choose *Run All*. The solvent comparison takes about a minute and the produced-water process a
few minutes. To run one without opening it:

```sh
jupyter nbconvert --to notebook --execute --inplace lithium_produced_water_process.ipynb
```

**As an app.** `voila lithium_produced_water_process.ipynb` (after `pip install voila`) serves a notebook as a web page
that shows only the text, widgets and results, with the code hidden; it runs every cell when the page opens.

The helper functions the notebooks call are the `lithium_*_tools.py` files beside them. The notebooks were checked on
epcsaft 1.2.0 (Engine commit `4cb87490`); another Engine version is refused.
