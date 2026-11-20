"""Shared helpers for the lithium solvent-extraction tutorial.

Each function takes the flowsheet model `m` (or its results) and does one piece of work that would hide the
flowsheet steps if it were written out in the notebook.
"""

import os
import sys
import tempfile
import textwrap
from importlib.metadata import version
from pathlib import Path

# One BLAS/OpenMP thread for the Engine, in this kernel and in the IPOPT processes it starts: the conda Engine links a
# threaded OpenBLAS, whose threads would use up IPOPT's CPU-time limit on these small equilibrium problems.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from idaes.core import FlowDirection, FlowsheetBlock  # noqa: E402
from prommis.solvent_extraction.solvent_extraction import (
    SolventExtraction,
)  # noqa: E402
from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    ExternalFunction,
    NonNegativeReals,
    Objective,
    Param,
)  # noqa: E402
from pyomo.environ import RangeSet, Reference, Var  # noqa: E402


from prommis.lithium_extraction.paths import REPO_ROOT as ROOT
from prommis.lithium_extraction.extraction import stage
from prommis.lithium_extraction.process.flowsheet import installed_engine
from prommis.lithium_extraction.extraction.stage import (
    ExtentReactions,
    MolarFlowParameters,
)

PHASES = ("aqueous", "organic")


def number(x):
    """Three significant figures; whole numbers with thousands separators from 1000 up; just below 1 stays below 1."""
    if 0.999 < x < 1:
        return "> 0.999"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


pd.set_option("display.float_format", number)


def shown(table, missing):
    """A table as text: numbers by `number`, empty cells as `missing`."""
    return table.map(lambda x: missing if pd.isna(x) else number(x))


# Display names of the HBTA/TOPO organic complexes; the model's species names stay unchanged.
LABELS = {"lithium-carrier": "LiA·T complex", "sodium-carrier": "NaA·T complex"}
KEYS = {
    shown_as: name for name, shown_as in LABELS.items()
}  # the notebook names the complexes by their labels


def label(species):
    return LABELS.get(species, species)


def versions():
    """The software this run uses, with the installed Engine."""
    engine = installed_engine()
    return pd.Series(
        {
            "ePC-SAFT Engine": engine["filename"].split("-")[1],
            **{
                name: version(package).split("+")[
                    0
                ]  # release number, without a local build tag
                for name, package in (
                    ("IDAES", "idaes-pse"),
                    ("PrOMMiS", "prommis"),
                    ("Pyomo", "pyomo"),
                )
            },
        },
        name="version",
    ).to_frame()


def log_directory(name):
    """A fresh scratch directory for solver logs."""
    return Path(tempfile.mkdtemp(prefix=f"{name}_"))


# ---------------------------------------------------------------- building the extraction stages


def species(chemistry):
    """The species of each phase in a compiled Engine record."""
    return {
        f"{phase} species": [label(s) for s in support]
        for phase, support in zip(PHASES, chemistry.supports)
    }


def extent_stoichiometry(chemistry):
    """Reaction-package stoichiometry: one transfer extent per independent species.

    One pivot species is chosen per conserved total (rows of C); every other species gets its own extent and the
    pivots change by whatever keeps the totals fixed, so the extent directions N satisfy C N = 0.
    """
    species, C = chemistry.species, np.asarray(chemistry.conservation, dtype=float)
    pivots, independent = _split(chemistry)
    N = np.zeros((len(species), len(independent)))
    N[pivots] = -np.linalg.solve(C[:, pivots], C[:, independent])
    N[independent, range(len(independent))] = 1.0
    assert np.abs(C @ N).max() < 1e-12, "an extent would change a conserved total"
    phase = {
        s: PHASES[p] for p, support in enumerate(chemistry.supports) for s in support
    }
    return {
        (species[j], phase[s], s): float(N[i, r])
        for r, j in enumerate(independent)
        for i, s in enumerate(species)
        if N[i, r]
    }


def label_streams(m, feed_l_per_raw_l=1.0):
    """Short names for the extraction streams, and nonnegative flows.

    `m.fs.boundary[s]` is the feed of species s (brine or fresh organic) and `m.fs.stage[k].outlet[s]` the amount of s
    leaving stage k in its own phase. `feed_l_per_raw_l` is litres of extraction feed per litre of raw water.
    """
    unit = m.fs.extraction.mscontactor
    stages = len(unit.elements)
    m.fs.stages = RangeSet(stages)
    m.fs.washed_feed_l_per_raw_l = Param(initialize=feed_l_per_raw_l, mutable=True)
    phase = {s: "aqueous" for s in m.fs.aqueous_properties.component_list}
    phase |= {s: "organic" for s in m.fs.organic_properties.component_list}
    outlets = {
        (k, s): getattr(unit, p)[0, k].flow_mol_comp[s]
        for k in m.fs.stages
        for s, p in phase.items()
    }
    feeds = {
        s: getattr(unit, p + "_inlet_state")[0].flow_mol_comp[s]
        for s, p in phase.items()
    }
    for flow in (*outlets.values(), *feeds.values()):
        flow.setlb(0.0)
    m.fs.boundary = Reference(feeds)
    m.fs.stage = Block(m.fs.stages)
    for k in m.fs.stages:
        m.fs.stage[k].outlet = Reference({s: outlets[k, s] for s in phase})


def add_stage_equilibrium(m, chemistries):
    """Close every stage with the Engine: n_out[s, k] = E_s(n_in[k]) for each independent species s.

    `chemistries` holds one compiled Engine record per stage. The Engine is a Pyomo ExternalFunction with exact first
    and second derivatives; its arguments are the record's fixed T [K] and P [Pa], then the amounts entering the stage
    (brine from stage k-1, organic from stage k+1, or the feeds) and any fixed stage parameters.
    """
    first = chemistries[0]
    species = first.species
    phase = {s: p for p, support in enumerate(first.supports) for s in support}
    parameters = dict(first.parameters)
    if parameters:
        m.fs.parameters = Var(
            tuple(parameters), domain=NonNegativeReals, initialize=parameters
        )
        m.fs.parameters.fix()
    m.fs.engine = ExternalFunction(
        library=str(first.library),
        function="epcsaft_equilibrium_phase_component_amount",
    )

    def incoming(k, s):
        if s in parameters:
            return m.fs.parameters[s]
        other = k - 1 if phase[s] == 0 else k + 1
        return m.fs.stage[other].outlet[s] if other in m.fs.stages else m.fs.boundary[s]

    independent = _split(first)[1]
    for (
        k
    ) in (
        m.fs.stages
    ):  # indexed by species position j, as the repository's stage report reads it
        c = chemistries[k - 1]
        names = c.record_species or species
        for s in set(species) - set(names):
            m.fs.stage[k].outlet[s].fix(0.0)
        m.fs.stage[k].equilibrium = Constraint(
            [j for j in independent if species[j] in names],
            rule=lambda b, j, k=k, c=c, names=names: b.outlet[species[j]]
            == m.fs.engine(
                c.temperature_k,
                c.pressure_pa,
                *(incoming(k, q) for q in c.control_species),
                c.record,
                str(phase[species[j]]),
                str(names.index(species[j])),
            ),
        )
    m.feasibility = Objective(expr=0.0)  # a square simulation: a constant objective
    m._chemistry, m._chemistries, m._incoming = first, tuple(chemistries), incoming


def extraction_flowsheet(chemistries, feed_l_per_raw_l=1.0):
    """The build of Step 4 as one call, for one compiled Engine record per stage: model and flowsheet, the two
    property packages, the transfer extents, the PrOMMiS SolventExtraction unit, the stream names and the Engine
    equilibrium in every stage."""
    first = chemistries[0]
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.aqueous_properties = MolarFlowParameters(
        phase="aqueous", species=first.supports[0]
    )
    m.fs.organic_properties = MolarFlowParameters(
        phase="organic", species=first.supports[1]
    )
    m.fs.extents = ExtentReactions(stoichiometry=extent_stoichiometry(first))
    m.fs.extraction = SolventExtraction(
        number_of_finite_elements=len(chemistries),
        heterogeneous_reaction_package=m.fs.extents,
        aqueous_stream={
            "property_package": m.fs.aqueous_properties,
            "flow_direction": FlowDirection.forward,
        },
        organic_stream={
            "property_package": m.fs.organic_properties,
            "flow_direction": FlowDirection.backward,
        },
    )
    label_streams(m, feed_l_per_raw_l)
    add_stage_equilibrium(m, chemistries)
    return m


def solve(m, name):
    """IPOPT with exact second derivatives and tight tolerances; stops with an error unless the optimality conditions
    in its log meet their limits."""
    kkt = stage.solve(m, log_directory(name))["kkt"]
    print(
        f"IPOPT: optimal; largest optimality-condition residual {max(kkt.values()):.0e}"
    )


def _split(chemistry):
    """(pivot, independent) species indices: one pivot per conserved total, chosen in species order."""
    C, pivots = np.asarray(chemistry.conservation, dtype=float), []
    for j in range(len(chemistry.species)):
        if np.linalg.matrix_rank(C[:, [*pivots, j]]) > len(pivots):
            pivots.append(j)
    return pivots, [j for j in range(len(chemistry.species)) if j not in pivots]


def stage_profile(m, species):
    """Amount of each species leaving each stage, from the model (mol per litre of extraction feed)."""
    keys = [KEYS.get(s, s) for s in species]
    return pd.DataFrame(
        {
            k: {label(s): m.fs.stage[k].outlet[s].value for s in keys}
            for k in m.fs.stages
        }
    ).rename_axis(columns="stage")


# ---------------------------------------------------------------- showing the reproduction check


def plot_checks(checks, group, title):
    """Every comparison as |reproduced - published| / tolerance; a point below 1 is within its tolerance.

    `checks` needs the columns `group`, 'kind', 'difference', 'limit' and 'passed'. Differences of 1e-12 of the
    tolerance or less, exact agreement included, are drawn at the bottom; text fields compared for equality are
    counted in the table but not plotted.
    """
    floor = 1e-12
    numeric = (
        pd.to_numeric(checks["limit"], errors="coerce").notna()
        & checks["difference"].notna()
    )
    difference, limit = checks.loc[numeric, "difference"], checks.loc[
        numeric, "limit"
    ].astype(float)
    ratio = (difference / limit.where(limit > 0)).fillna(
        pd.Series(np.where(difference > 0, np.inf, 0.0), difference.index)
    )
    ratio = ratio.clip(lower=floor)  # a zero limit means exact agreement is required
    groups, kinds = list(dict.fromkeys(checks[group])), list(
        dict.fromkeys(checks.loc[numeric, "kind"])
    )
    colors = dict(zip(kinds, plt.get_cmap("tab10").colors))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for kind in kinds:
        rows = numeric & (checks["kind"] == kind)
        x = checks.loc[rows, group].map(groups.index) + rng.uniform(
            -0.25, 0.25, rows.sum()
        )
        ax.scatter(
            x, ratio[rows[rows].index], s=16, alpha=0.75, color=colors[kind], label=kind
        )
    ax.axhline(1, color="#a23b2a", linewidth=1.2)
    ax.text(
        -0.45, 1.5, "tolerance", color="#a23b2a", ha="left", va="bottom", fontsize=9
    )
    ticks = [floor, *10.0 ** np.arange(-10, 1, 2)]
    ax.set(
        yscale="log",
        ylim=(floor / 3, 30),
        yticks=ticks,
        xticks=range(len(groups)),
        yticklabels=[r"$\leq 10^{-12}$", *(f"$10^{{{e}}}$" for e in range(-10, 1, 2))],
        xticklabels=[textwrap.fill(g, 16) for g in groups],
        xlim=(-0.5, len(groups) - 0.5),
        ylabel="|difference| / tolerance",
        title=title,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    plt.show()
    by_group = checks.groupby(group, sort=False)
    return pd.DataFrame(
        {
            "comparisons": by_group.size(),
            "largest difference / tolerance": ratio.groupby(
                checks.loc[numeric, group], sort=False
            )
            .max()
            .map(lambda r: f"{r:.1e}"),
            "all within tolerance": by_group["passed"]
            .all()
            .map({True: "✓", False: "✗"}),
        },
        index=groups,
    )
