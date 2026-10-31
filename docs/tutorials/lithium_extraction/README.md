# Lithium solvent-extraction tutorials

Two notebooks that build lithium solvent-extraction flowsheets with IDAES and PrOMMiS, closing every stage with
ePC-SAFT phase and chemical equilibrium from the [epcsaft](https://github.com/tannerpolley/epcsaft) Engine:

- `lithium_solvent_comparison.ipynb`: a three-stage countercurrent extraction of softened Smackover MS-2 brine,
  comparing four solvent systems, with the Engine's derivatives shown inside the model and a design optimization.
- `lithium_produced_water_process.ipynb`: a whole process from raw produced water to lithium carbonate (softening,
  extraction, scrub, strip, acid recovery, carbonation and operating cost) for two waters.

Each notebook checks every number it reproduces against the retained results of the Lithium_Extraction project and
stops with an error if one is outside its limit.

## Setup (64-bit Windows or Linux)

The Engine's conda packages exist for Python 3.13 only, so make a new environment; an existing PrOMMiS environment on
Python 3.11 cannot take them. From a Miniforge prompt, in the root of this PrOMMiS checkout:

```sh
conda create -n prommis-lithium -c conda-forge -c tannerwpolley python=3.13 epcsaft=1.2.0 reaktoro=2.13.0
conda env config vars set -n prommis-lithium PYTHONUTF8=1
conda activate prommis-lithium
pip install -e . "watertap>=1.8,<1.9" "ipywidgets>=8,<9" jupyterlab
idaes get-extensions
```

The Miniforge installer refuses a folder whose path contains a space; if your Windows user name has one, install
Miniforge to `C:\miniforge3`. Clone the repository with git rather than downloading a zip: PrOMMiS reads its version
from the git history.

`PYTHONUTF8=1` makes Python read the retained data files as UTF-8 on Windows. `tannerwpolley` is an interim channel for the Engine until its conda-forge package is published; after that,
`-c conda-forge` alone is enough. Reaktoro solves the softening step of the produced-water notebook.

## Run

```sh
cd docs/tutorials/lithium_extraction
jupyter lab
```

Open a notebook and choose *Run All*. The solvent comparison takes about a minute and the produced-water process a
few minutes.

## What is in this folder

- `tutorial_tools/`: the helper functions the notebooks call.
- `lithium_extraction/` and `analyses/`: the parts of the Lithium_Extraction project's model code that the notebooks
  import, under their original module names.
- `data/` and the `inputs/` and `results/` folders under `analyses/`: the source data, fitted parameters and retained
  results that the code reads and the checks compare against.

The notebooks were checked on epcsaft 1.2.0 (Engine commit `4cb87490`); another Engine version is refused.
