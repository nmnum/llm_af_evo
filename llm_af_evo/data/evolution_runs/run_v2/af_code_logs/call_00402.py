def score_pool(context):
    """
    Pure exploitation: rank by predicted objective sum only. Iterates over
    context["objective_names"] rather than hardcoding Tm/kD/viscosity, so
    this works on any oracle's objective set. No sign-flipping is applied
    here — gp[name]["mean"] is ALREADY in all-maximise convention by the
    time score_pool sees it (upstream strategy code does that conversion
    before building context), so re-applying objective_directions here
    would double-flip and invert behaviour on any min-objective.
    """
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        scores.append(sum(gp[name]["mean"] for name in names))
    return scores