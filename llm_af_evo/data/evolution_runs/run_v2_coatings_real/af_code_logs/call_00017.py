def score_pool(context):
    """Exploitation-plus-uncertainty with progress-aware weighting and normalized uncertainty bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Weight exploitation vs uncertainty: early = more UCB, late = more exploitation
        weight_exploit = 0.3 + 0.7 * (1 - progress)
        weight_uncertainty = 1.0 - weight_exploit
        scores.append(weight_exploit * mu_sum + weight_uncertainty * 2.0 * sigma_norm)
    return scores