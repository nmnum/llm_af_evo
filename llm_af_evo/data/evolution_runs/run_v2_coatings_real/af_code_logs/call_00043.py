def score_pool(context):
    """Exploitation with progressive uncertainty bonus, balancing objective prediction and exploration based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Start with exploitation, shift towards exploration as progress increases
        weight = 1.0 + 3.0 * (progress - 0.5)  # Shifts from 1.0 to 2.5 as progress goes from 0.5 to 1.0
        scores.append(mu_sum + max(0.0, weight) * sigma_norm)
    return scores