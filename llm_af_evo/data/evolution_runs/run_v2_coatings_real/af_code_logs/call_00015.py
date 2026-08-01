def score_pool(context):
    """Exploitation with progress-adaptive uncertainty bonus, balancing between greedy objective sum and UCB-style exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        weight = 1.0 + 3.0 * (1.0 - progress)
        scores.append(mu_sum + weight * sigma_norm)
    return scores