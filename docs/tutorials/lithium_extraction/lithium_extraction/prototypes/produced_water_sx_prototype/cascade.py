"""Shared #176 Gauss–Seidel stage sweep; chemistry and strict states belong to the caller."""


def sweep_stages(stage_count, feed, fresh, solve_stage, *, start=None, progress=None,
                 tolerance=1e-11, max_sweeps=1000, resume_sweeps=0):
    """Aqueous enters stage 1, fresh organic enters N; unchanged #166 stopping rule.

    ``solve_stage`` returns aqueous, organic, evidence row and composition guess.
    A progress callback retains the completed sweep for bounded resumable units.
    """
    if not 0 <= resume_sweeps < max_sweeps:
        raise RuntimeError('Resumed cascade has exhausted the unchanged sweep budget')
    organic = [dict(o) for o in start['organic']] if start else [dict(fresh) for _ in range(stage_count)]
    guesses = list(start['guesses']) if start else [None] * stage_count
    for sweep in range(resume_sweeps + 1, max_sweeps + 1):
        aqueous, rows, leaving = dict(feed), [], []
        for k in range(stage_count):
            entering = organic[k + 1] if k + 1 < stage_count else fresh
            aqueous, out_org, row, guesses[k] = solve_stage(k, aqueous, entering, guesses[k])
            rows.append(row)
            leaving.append(out_org)
        change = max(abs(a[x] - b[x]) for a, b in zip(leaving, organic) for x in a)
        organic = leaving
        state = {'organic': organic, 'guesses': guesses}
        if progress:
            progress(sweep, change, state)
        if change <= tolerance:
            break
    else:
        raise RuntimeError(f'cascade did not converge in {max_sweeps} sweeps (last change {change:g} mol/L)')
    return {'sweeps': sweep, 'last_change_mol_l': change, 'stages': rows, 'raffinate': aqueous,
            'loaded_organic': organic[0], 'start': state}
