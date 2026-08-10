def score_pool(context):
    """Exploitation with uncertainty-aware ranking: sum of means plus scaled std penalty, modulated by progress towards target."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Progress-aware scaling: early on, emphasise uncertainty; later, exploit more.
        progress = context["campaign"]["progress"]
        weight_factor = 0.5 + (2.0 * max(0, progress - 0.3))  # Start at 0.5 and ramp up to 2.5
        scores.append(mu_sum - weight_factor * sigma_norm)
    return scores