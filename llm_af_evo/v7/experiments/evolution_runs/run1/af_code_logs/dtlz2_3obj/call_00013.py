def modifier(context):
    """Combines progress decay and stagnation scaling for uncertainty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    weight = 0.3 * (1.0 - progress) * min(stagnant_batches / 5.0, 1.0)
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        values.append(weight * sigma_norm)
    return values