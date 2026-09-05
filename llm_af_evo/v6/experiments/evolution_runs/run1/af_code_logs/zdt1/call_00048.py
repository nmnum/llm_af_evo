def modifier(context):
    """Adaptive uncertainty bonus scaling with stagnant batches."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    stagnation_factor = min(context["campaign"]["stagnant_batches"] / 5.0, 1.0)
    base_weight = 0.3
    weight = base_weight * stagnation_factor
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        values.append(weight * sigma_norm)
    return values