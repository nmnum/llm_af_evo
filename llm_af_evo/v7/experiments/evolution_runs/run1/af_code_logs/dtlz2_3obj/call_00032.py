def modifier(context):
    """Uncertainty bonus scaled by stagnation level, encouraging exploration during stagnant campaigns."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    base_weight = 0.3522
    scaling_factor = min(stagnant_batches / 5.0, 1.0)
    weight = base_weight * scaling_factor
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        values.append(weight * sigma_norm)
    return values