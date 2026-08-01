def score_pool(context):
    """Exploitation-plus-uncertainty with progress-aware weighting and normalized uncertainty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Weight uncertainty bonus inversely to progress (less exploration late)
        weight = 1.0 - progress
        scores.append(mu_sum + weight * sigma_norm)
    return scores