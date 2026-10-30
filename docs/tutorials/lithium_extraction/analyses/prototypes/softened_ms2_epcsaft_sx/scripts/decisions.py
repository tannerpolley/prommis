"""#216 extraction controls in physical units; fixed until F2 passes."""

from pyomo.environ import Block, Constraint, Expression, Var


def attach(model, arm, *, release=False, target=None):
    fs, design = model.fs, model._design
    fs.paper1 = Block()
    b = fs.paper1
    d2 = arm.startswith('D2')
    names = ('O_A', 'B', 'S') if d2 else ('organic_to_aqueous_ratio', 'extractant_mol_l', 'saponification_ratio')
    bounds = ((.5, 2.), (.25, 1.), (.9, 1.5)) if d2 else (
        (.5, 3.), (.05, .30) if 'acid' in arm else (.1, .4), (0., 0.) if 'acid' in arm else (.5, 1.))
    model._decisions = {}
    for name, limits in zip(names, bounds):
        b.add_component(name, Var(initialize=design[name], bounds=limits))
        variable = getattr(b, name)
        variable.fix(design[name])
        if release and limits[0] != limits[1]:
            variable.unfix()
        model._decisions[name] = variable
    ratio, concentration, preparation = (model._decisions[n] for n in names)
    inlet = fs.boundary
    inlet['diluent'].unfix()
    b.diluent = Constraint(expr=inlet['diluent'] == ratio * model._feeds[1]['diluent'] / design[names[0]])
    if d2:
        for species, expression in (('naa', preparation * design['site_basis_li_mol_l']),
                                     ('buphen', ratio * concentration)):
            inlet[species].unfix()
            b.add_component(species, Constraint(expr=inlet[species] == expression))
        for name, species in (('sites_total_mol', 'naa'), ('buphen_total_mol', 'buphen')):
            fs.parameters[name].unfix()
            b.add_component(name, Constraint(expr=fs.parameters[name] == inlet[species]))
        b.site_min = Constraint(expr=inlet['naa'] >= .020 * ratio)
        b.site_max = Constraint(expr=inlet['naa'] <= .060 * ratio)
    else:
        for species, expression in (('hbta', ratio * concentration * (1 - preparation)),
                                    ('sodium-carrier', ratio * concentration * preparation),
                                    ('topo', ratio * concentration * (1 - preparation))):
            inlet[species].unfix()
            b.add_component(species, Constraint(expr=inlet[species] == expression))
        if 'topo_total_mol' in fs.parameters:
            fs.parameters['extractant_mol_l'].unfix()
            b.concentration = Constraint(expr=fs.parameters['extractant_mol_l'] == concentration)
            fs.topo_quantity.set_value(fs.parameters['topo_total_mol'] == ratio * concentration)
        b.loading = Constraint(fs.stages, rule=lambda _, k:
            fs.stage[k].outlet['lithium-carrier'] <= .72 * ratio * concentration)
    b.extraction = Expression(expr=1 - fs.stage[len(fs.stages)].outlet['lithium-cation'] /
                             model._feeds[0]['lithium-cation'])
    if target is not None:
        b.target = Constraint(expr=b.extraction >= target)
