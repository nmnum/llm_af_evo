def score_pool(context):
    """Adaptive exploitation and uncertainty tradeoff: dynamic weighting of means and uncertainty based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: early progress favors uncertainty, later progress favors exploitation
        w = 0.5 + 0.5 * (1 - progress)  # decreases from 1 to 0 as progress goes from 0 to 1
        scores.append(w * mu_sum + (1 - w) * sigma_norm)
    return scores