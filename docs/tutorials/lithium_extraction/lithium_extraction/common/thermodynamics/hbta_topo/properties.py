"""Minimal IDAES state blocks for the HBTA/TOPO circuit.

The packages are material-balance packages, not thermodynamic activity models.
They keep every reported species and solvent inventory on an explicit molar
flow basis while the source-bounded transfer closures live in ``circuit.py``.
"""

from __future__ import annotations

from idaes.core import (
    Component,
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars
from pyomo.environ import Param, Set, Var, units


@declare_process_block_class("HbtaTopoAqueousParameters")
class HbtaTopoAqueousParameterData(PhysicalParameterBlock):
    """Charge-explicit Li/Na/Cl/H/OH aqueous material-balance package."""

    def build(self):
        super().build()
        self.liquid = Phase()
        self.H2O = Component()
        self.Li = Component()
        self.Na = Component()
        self.Cl = Component()
        self.H = Component()
        self.OH = Component()
        self.dissolved_components = Set(initialize=["Li", "Na", "Cl", "H", "OH"])
        self.mw = Param(
            self.component_list,
            units=units.kg / units.mol,
            initialize={
                "H2O": 18.01528e-3,
                "Li": 6.94e-3,
                "Na": 22.98976928e-3,
                "Cl": 35.45e-3,
                "H": 1.00784e-3,
                "OH": 17.00734e-3,
            },
        )
        self.dens_mass = Param(
            initialize=1.0,
            units=units.kg / units.litre,
            mutable=True,
            doc="Screening aqueous density; no activity or volume-change claim.",
        )
        self._state_block_class = HbtaTopoAqueousStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(
            {
                "time": units.hour,
                "mass": units.kg,
                "amount": units.mol,
                "length": units.m,
                "temperature": units.K,
            }
        )


class _HbtaTopoAqueousStateBlock(StateBlock):
    def fix_initialization_states(self):
        fix_state_vars(self)


@declare_process_block_class(
    "HbtaTopoAqueousStateBlock",
    block_class=_HbtaTopoAqueousStateBlock,
)
class HbtaTopoAqueousStateBlockData(StateBlockData):
    def build(self):
        super().build()
        self.conc_mol_comp = Var(
            self.params.dissolved_components,
            units=units.mol / units.L,
            initialize=1.0e-8,
            bounds=(0.0, None),
        )
        self.flow_vol = Var(units=units.L / units.hour, initialize=1.0, bounds=(1.0e-12, None))

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def get_material_flow_terms(self, _phase, component):
        if component == "H2O":
            return self.flow_vol * self.params.dens_mass / self.params.mw[component]
        return self.flow_vol * self.conc_mol_comp[component]

    def get_material_density_terms(self, _phase, component):
        if component == "H2O":
            return units.convert(
                self.params.dens_mass / self.params.mw[component],
                to_units=units.mol / units.m**3,
            )
        return units.convert(
            self.conc_mol_comp[component],
            to_units=units.mol / units.m**3,
        )

    def define_state_vars(self):
        return {"flow_vol": self.flow_vol, "conc_mol_comp": self.conc_mol_comp}


@declare_process_block_class("HbtaTopoOrganicParameters")
class HbtaTopoOrganicParameterData(PhysicalParameterBlock):
    """Analytical HBTA/TOPO/diluent and loaded-metal organic inventories."""

    def build(self):
        super().build()
        self.liquid = Phase()
        self.HBTA = Component()
        self.TOPO = Component()
        self.Diluent = Component()
        self.Li = Component()
        self.Na = Component()
        self.components_with_concentration = Set(initialize=["HBTA", "TOPO", "Diluent", "Li", "Na"])
        self._state_block_class = HbtaTopoOrganicStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(
            {
                "time": units.hour,
                "mass": units.kg,
                "amount": units.mol,
                "length": units.m,
                "temperature": units.K,
            }
        )


class _HbtaTopoOrganicStateBlock(StateBlock):
    def fix_initialization_states(self):
        fix_state_vars(self)


@declare_process_block_class(
    "HbtaTopoOrganicStateBlock",
    block_class=_HbtaTopoOrganicStateBlock,
)
class HbtaTopoOrganicStateBlockData(StateBlockData):
    def build(self):
        super().build()
        self.conc_mol_comp = Var(
            self.params.components_with_concentration,
            units=units.mol / units.L,
            initialize=1.0e-8,
            bounds=(0.0, None),
        )
        self.flow_vol = Var(units=units.L / units.hour, initialize=1.0, bounds=(1.0e-12, None))

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def get_material_flow_terms(self, _phase, component):
        return self.flow_vol * self.conc_mol_comp[component]

    def get_material_density_terms(self, _phase, component):
        return units.convert(
            self.conc_mol_comp[component],
            to_units=units.mol / units.m**3,
        )

    def define_state_vars(self):
        return {"flow_vol": self.flow_vol, "conc_mol_comp": self.conc_mol_comp}
