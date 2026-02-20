#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Initial property package for the aqueous phase solution of the solvent extraction
unit operation.

Authors: Arkoprabho Dasgupta

"""

from pyomo.environ import Param, Set, Var, units

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


@declare_process_block_class("LiILSolExAqParameters")
class LiILSolExAqParameterData(PhysicalParameterBlock):
    """
    This is a property package for the aqueous phase solution of the solvent extraction
    unit operation of Lithium extraction from magnesium-rich brines with ionic liquid-based extractants
    Yu 2023 https://doi.org/10.1016/j.ces.2023.119682

    This includes the following components:

    * Aqueous Solvent: H2O
    * Cations: Li, Mg
    * Anions: Cl

    """

    def build(self):
        super().build()

        self.liquid = Phase()

        # Aqueous Solvents
        self.H2O = Component()

        # Cations
        self.Li = Component()
        self.Mg = Component()

        # Anions
        self.Cl = Component()


        self.dissolved_elements = Set(
            initialize=[
                "Li",
                "Mg",
                "Cl",
            ]
        )

        self.mw = Param(
            self.component_list,
            units=units.kg / units.mol,
            initialize={
                "H2O": 18.015e-3,
                "Li": 6.94e-3,
                "Mg": 24.31e-3,
                "Cl": 35.45e-3,
            },
        )

        # density of H2O
        self.dens_mass = Param(
            initialize=1,
            units=units.kg / units.litre,
            mutable=True,
        )

        self._state_block_class = LiILSolExAqStateBlock

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


class _LiILSolExAqStateBlock(StateBlock):
    def fix_initialization_states(self):
        fix_state_vars(self)


@declare_process_block_class("LiILSolExAqStateBlock", block_class=_LiILSolExAqStateBlock)
class LiILSolExAqStateBlockData(StateBlockData):
    """
    State block for aqueous phase solution of the solvent extraction process.

    """

    def build(self):
        super().build()

        self.conc_mass_comp = Var(
            self.params.dissolved_elements,
            units=units.mg / units.L,
            bounds=(1e-20, None),
        )

        self.flow_vol = Var(units=units.L / units.hour, bounds=(1e-8, None))

        self.conc_mol_comp = Var(
            self.params.dissolved_elements,
            units=units.mol / units.L,
            initialize=1e-5,
            bounds=(1e-20, None),
        )

        # Concentration conversion constraint
        @self.Constraint(self.params.dissolved_elements)
        def molar_concentration_constraint(b, j):
            return (
                units.convert(
                    b.conc_mol_comp[j] * b.params.mw[j], to_units=units.mg / units.litre
                )
                == b.conc_mass_comp[j]
            )

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def get_material_flow_terms(self, p, j):
        if j == "H2O":
            return self.flow_vol * self.params.dens_mass / self.params.mw[j]
        else:
            return units.convert(
                self.flow_vol * self.conc_mass_comp[j] / self.params.mw[j],
                to_units=units.mol / units.hour,
            )

    def get_material_density_terms(self, p, j):
        if j == "H2O":
            return units.convert(
                self.params.dens_mass / self.params.mw[j],
                to_units=units.mol / units.m**3,
            )
        else:
            return units.convert(
                self.conc_mass_comp[j] / self.params.mw[j],
                to_units=units.mol / units.m**3,
            )

    def define_state_vars(self):
        return {"flow_vol": self.flow_vol, "conc_mass_comp": self.conc_mass_comp}
