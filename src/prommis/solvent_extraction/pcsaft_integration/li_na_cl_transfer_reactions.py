#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Reaction package shim for Li/Na/Cl solvent-extraction examples.

The Li/Na/Cl PC-SAFT integration constrains transfer directly on the shared-component
MSContactor material transfer terms. The current SolventExtraction wrapper still expects
a heterogeneous reaction package so its distribution-ratio constraints can be indexed.
This package intentionally declares no heterogeneous reactions or elements, leaving the
explicit Li/Na/Cl transfer constraints as the active transfer model.
"""

from pyomo.common.config import ConfigValue
from pyomo.environ import Set, units

from idaes.core import ProcessBlock, ProcessBlockData, declare_process_block_class
from idaes.core.base import property_meta
from idaes.core.util.misc import add_object_reference


@declare_process_block_class("LiNaClTransferReactions")
class LiNaClTransferReactionsData(
    ProcessBlockData, property_meta.HasPropertyClassMetadata
):
    def build(self):
        super().build()

        self._reaction_block_class = LiNaClTransferReactionBlock
        self.element_list = Set(initialize=[])
        self.reaction_idx = Set(initialize=[])
        self.reaction_stoichiometry = {}

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(
            {
                "time": units.hour,
                "length": units.m,
                "mass": units.kg,
                "amount": units.mol,
                "temperature": units.K,
            }
        )

    @property
    def reaction_block_class(self):
        if self._reaction_block_class is None:
            raise AttributeError(
                f"{self.name} has not assigned a ReactionBlock class."
            )
        return self._reaction_block_class

    def build_reaction_block(self, *args, **kwargs):
        default = kwargs.pop("default", {})
        initialize = kwargs.pop("initialize", {})

        if initialize == {}:
            default["parameters"] = self
        else:
            for i in initialize:
                initialize[i]["parameters"] = self

        return self.reaction_block_class(
            *args, **kwargs, **default, initialize=initialize
        )


class _LiNaClTransferReactionBlock(ProcessBlock):
    pass


@declare_process_block_class(
    "LiNaClTransferReactionBlock", block_class=_LiNaClTransferReactionBlock
)
class LiNaClTransferReactionBlockData(ProcessBlockData):
    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare(
        "parameters",
        ConfigValue(
            description="Li/Na/Cl transfer reaction parameter block.",
        ),
    )

    def build(self):
        super().build()
        add_object_reference(self, "_params", self.config.parameters)

    @property
    def params(self):
        return self._params
