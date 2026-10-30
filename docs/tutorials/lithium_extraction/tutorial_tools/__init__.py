"""Shared helpers for the lithium solvent-extraction tutorials.

Each function takes the flowsheet model `m` (or its results) and does one piece of work that would hide the
flowsheet steps if it were written out in the notebook. Importing this package puts the tutorial folder on
`sys.path` and makes it the working directory, because the Lithium modules read their data files relative to it.
"""

# ruff: noqa: E402
import os

# One BLAS/OpenMP thread for the Engine, in this kernel and in the IPOPT processes it starts: the conda Engine links a
# threaded OpenBLAS, whose threads would use up IPOPT's CPU-time limit on these small equilibrium problems.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import tempfile
import textwrap
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from idaes.core import UnitModelBlockData
from pyomo.environ import Block, Constraint, ExternalFunction, NonNegativeReals, Objective, Param, RangeSet
from pyomo.environ import Reference, Var, value
from pyomo.network import Arc, Port

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from lithium_extraction.prototypes.produced_water_sx_prototype.flowsheet import installed_engine  # noqa: E402

PHASES = ('aqueous', 'organic')


def number(x):
    """Three significant figures; whole numbers with thousands separators from 1000 up; just below 1 stays below 1."""
    if .999 < x < 1:
        return '> 0.999'
    return f'{x:,.0f}' if abs(x) >= 1000 else f'{x:.3g}'


pd.set_option('display.float_format', number)


def versions():
    """The software this run uses, with the pinned Engine wheel."""
    engine = installed_engine()
    return pd.Series({'ePC-SAFT Engine': f"{engine['filename'].split('-')[1]} (source commit {engine['source_commit'][:8]})",
                      'IDAES': version('idaes-pse'), 'PrOMMiS': version('prommis'), 'Pyomo': version('pyomo')},
                     name='version').to_frame()


def log_directory(name):
    """A fresh scratch directory for solver logs."""
    return Path(tempfile.mkdtemp(prefix=f'{name}_'))


# ---------------------------------------------------------------- building the extraction stages

def extent_stoichiometry(chemistry):
    """Reaction-package stoichiometry: one transfer extent per independent species.

    One pivot species is chosen per conserved total (rows of C); every other species gets its own extent and the
    pivots change by whatever keeps the totals fixed, so the extent directions N satisfy C N = 0.
    """
    species, C = chemistry.species, np.asarray(chemistry.conservation, dtype=float)
    pivots, independent = _split(chemistry)
    N = np.zeros((len(species), len(independent)))
    N[pivots] = -np.linalg.solve(C[:, pivots], C[:, independent])
    N[independent, range(len(independent))] = 1.
    assert np.abs(C @ N).max() < 1e-12, 'an extent would change a conserved total'
    phase = {s: PHASES[p] for p, support in enumerate(chemistry.supports) for s in support}
    return {(species[j], phase[s], s): float(N[i, r])
            for r, j in enumerate(independent) for i, s in enumerate(species) if N[i, r]}


def label_streams(m, feed_l_per_raw_l=1.):
    """Short names for the extraction streams, and nonnegative flows.

    `m.fs.boundary[s]` is the feed of species s (brine or fresh organic) and `m.fs.stage[k].outlet[s]` the amount of s
    leaving stage k in its own phase. `feed_l_per_raw_l` is litres of extraction feed per litre of raw water.
    """
    unit = m.fs.extraction.mscontactor
    stages = len(unit.elements)
    m.fs.stages = RangeSet(stages)
    m.fs.washed_feed_l_per_raw_l = Param(initialize=feed_l_per_raw_l, mutable=True)
    phase = {s: 'aqueous' for s in m.fs.aqueous_properties.component_list}
    phase |= {s: 'organic' for s in m.fs.organic_properties.component_list}
    outlets = {(k, s): getattr(unit, p)[0, k].flow_mol_comp[s] for k in m.fs.stages for s, p in phase.items()}
    feeds = {s: getattr(unit, p + '_inlet_state')[0].flow_mol_comp[s] for s, p in phase.items()}
    for flow in (*outlets.values(), *feeds.values()):
        flow.setlb(0.)
    m.fs.boundary = Reference(feeds)
    m.fs.stage = Block(m.fs.stages)
    for k in m.fs.stages:
        m.fs.stage[k].outlet = Reference({s: outlets[k, s] for s in phase})


def add_stage_equilibrium(m, chemistries):
    """Close every stage with the Engine: n_out[s, k] = E_s(n_in[k]) for each independent species s.

    `chemistries` holds one compiled Engine record per stage. The Engine is a Pyomo ExternalFunction with exact first
    and second derivatives; its arguments are the amounts entering the stage (brine from stage k-1, organic from
    stage k+1, or the feeds) and any fixed stage parameters.
    """
    first = chemistries[0]
    species = first.species
    phase = {s: p for p, support in enumerate(first.supports) for s in support}
    parameters = dict(first.parameters)
    if parameters:
        m.fs.parameters = Var(tuple(parameters), domain=NonNegativeReals, initialize=parameters)
        m.fs.parameters.fix()
    m.fs.engine = ExternalFunction(library=str(first.library), function='epcsaft_equilibrium_phase_component_amount')

    def incoming(k, s):
        if s in parameters:
            return m.fs.parameters[s]
        other = k - 1 if phase[s] == 0 else k + 1
        return m.fs.stage[other].outlet[s] if other in m.fs.stages else m.fs.boundary[s]

    independent = _split(first)[1]
    for k in m.fs.stages:  # indexed by species position j, as the repository's stage report reads it
        c = chemistries[k - 1]
        names = c.record_species or species
        for s in set(species) - set(names):
            m.fs.stage[k].outlet[s].fix(0.)
        m.fs.stage[k].equilibrium = Constraint([j for j in independent if species[j] in names],
            rule=lambda b, j, k=k, c=c, names=names: b.outlet[species[j]] == m.fs.engine(
                *(incoming(k, q) for q in c.control_species), c.record, str(phase[species[j]]),
                str(names.index(species[j]))))
    m.feasibility = Objective(expr=0.)  # a square simulation: a constant objective
    m._chemistry, m._chemistries, m._incoming = first, tuple(chemistries), incoming


def _split(chemistry):
    """(pivot, independent) species indices: one pivot per conserved total, chosen in species order."""
    C, pivots = np.asarray(chemistry.conservation, dtype=float), []
    for j in range(len(chemistry.species)):
        if np.linalg.matrix_rank(C[:, [*pivots, j]]) > len(pivots):
            pivots.append(j)
    return pivots, [j for j in range(len(chemistry.species)) if j not in pivots]


def _engine_arguments(m, k, s):
    """Stage k's Engine call for species s: amounts entering the stage, then the record, phase and position."""
    c = m._chemistries[k - 1]
    names = c.record_species or c.species
    phase = next(p for p, support in enumerate(c.supports) if s in support)
    return [value(m._incoming(k, q)) for q in c.control_species], [c.record, str(phase), str(names.index(s))]


def engine_check(m, k):
    """Each Engine-closed outlet of stage k: a direct Engine call on the solved inlet beside the flowsheet value."""
    rows = {}
    for j in m.fs.stage[k].equilibrium:
        s = m._chemistries[k - 1].species[j]
        inputs, record = _engine_arguments(m, k, s)
        direct, solved = m.fs.engine.evaluate([*inputs, *record]), value(m.fs.stage[k].outlet[s])
        rows[s] = {'direct Engine call (mol/L)': direct, 'flowsheet outlet (mol/L)': solved,
                   'difference (mol/L)': abs(direct - solved)}
    return pd.DataFrame(rows).T.rename_axis(f'leaving stage {k}')


def engine_derivatives(m, k, s):
    """The Engine's exact first derivatives of outlet s of stage k against central finite differences."""
    inputs, record = _engine_arguments(m, k, s)
    _, gradient, _ = m.fs.engine.evaluate_fgh([*inputs, *record])
    rows = {}
    for i, q in enumerate(m._chemistries[k - 1].control_species):
        h = 1e-6 * max(abs(inputs[i]), 1e-3)  # a step of at least 1e-9 mol/L keeps round-off below the truncation error
        up, down = list(inputs), list(inputs)
        up[i] += h
        down[i] -= h
        difference = (m.fs.engine.evaluate([*up, *record]) - m.fs.engine.evaluate([*down, *record])) / (2 * h)
        rows[f'entering {q}'] = {'exact (Engine)': gradient[i], 'finite difference': difference,
                                 'relative difference': abs(gradient[i] - difference) / max(abs(gradient[i]), 1e-12)}
    return pd.DataFrame(rows).T.rename_axis(f'd n(out, {s}) / d n(in)')


def stage_profile(m, species):
    """Amount of each species leaving each stage, from the model (mol per litre of extraction feed)."""
    return pd.DataFrame({k: {s: m.fs.stage[k].outlet[s].value for s in species} for k in m.fs.stages}
                        ).rename_axis(columns='stage')


# ---------------------------------------------------------------- inspecting a flowsheet

def unit_models(model):
    """Every IDAES unit model on the flowsheet: its name, class and ports, in build order."""
    rows = []
    for unit in model.fs.component_data_objects(descend_into=True):
        if isinstance(unit, UnitModelBlockData) and not isinstance(unit.parent_block(), UnitModelBlockData):
            ports = [p.local_name for p in unit.component_objects(Port, descend_into=False)]
            rows.append({'unit': unit.name.removeprefix('fs.'),
                         'model class': type(unit).__name__.removeprefix('_Scalar').removesuffix('Data'),
                         'ports': ', '.join(ports)})
    return pd.DataFrame(rows).set_index('unit')


def arcs(model):
    """Every Arc on the flowsheet: the outlet port it reads and the inlet port it feeds."""
    return pd.DataFrame([{'arc': a.name.removeprefix('fs.'), 'from': a.source.name.removeprefix('fs.'),
                          'to': a.destination.name.removeprefix('fs.')}
                         for a in model.fs.component_data_objects(Arc, descend_into=True)]).set_index('arc')


# ---------------------------------------------------------------- showing a retained-record check

def plot_checks(checks, group, title):
    """Every comparison as |reproduced - retained| / tolerance; a point below 1 is within its tolerance.

    `checks` needs the columns `group`, 'kind', 'difference', 'limit' and 'passed'. Differences of 1e-12 of the
    tolerance or less, exact agreement included, are drawn at the bottom; text fields compared for equality are
    counted in the table but not plotted.
    """
    floor = 1e-12
    numeric = pd.to_numeric(checks['limit'], errors='coerce').notna() & checks['difference'].notna()
    difference, limit = checks.loc[numeric, 'difference'], checks.loc[numeric, 'limit'].astype(float)
    ratio = (difference / limit.where(limit > 0)).fillna(pd.Series(np.where(difference > 0, np.inf, 0.), difference.index))
    ratio = ratio.clip(lower=floor)  # a zero limit means exact agreement is required
    groups, kinds = list(dict.fromkeys(checks[group])), list(dict.fromkeys(checks.loc[numeric, 'kind']))
    colors = dict(zip(kinds, plt.get_cmap('tab10').colors))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for kind in kinds:
        rows = numeric & (checks['kind'] == kind)
        x = checks.loc[rows, group].map(groups.index) + rng.uniform(-.25, .25, rows.sum())
        ax.scatter(x, ratio[rows[rows].index], s=16, alpha=.75, color=colors[kind], label=kind)
    ax.axhline(1, color='#a23b2a', linewidth=1.2)
    ax.text(-.45, 1.5, 'tolerance', color='#a23b2a', ha='left', va='bottom', fontsize=9)
    ticks = [floor, *10. ** np.arange(-10, 1, 2)]
    ax.set(yscale='log', ylim=(floor / 3, 30), yticks=ticks, xticks=range(len(groups)),
           yticklabels=[r'$\leq 10^{-12}$', *(f'$10^{{{e}}}$' for e in range(-10, 1, 2))],
           xticklabels=[textwrap.fill(g, 16) for g in groups], xlim=(-.5, len(groups) - .5),
           ylabel='|difference| / tolerance', title=title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    plt.show()
    by_group = checks.groupby(group, sort=False)
    return pd.DataFrame({'comparisons': by_group.size(),
                         'largest difference / tolerance': ratio.groupby(checks.loc[numeric, group], sort=False).max()
                         .map(lambda r: f'{r:.1e}'),
                         'all within tolerance': by_group['passed'].all().map({True: '✓', False: '✗'})}, index=groups)
