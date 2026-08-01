def score_pool(context):
    """Exploitation with dynamic weighting and uncertainty bonus, adapting to campaign progress."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: early progress favors uncertainty, later favors exploitation
        w_exploit = 1.0 - min(progress * 2.0, 1.0)
        w_uncert = min(progress * 2.0, 1.0)
        scores.append(w_exploit * mu_sum + w_uncert * sigma_sum)
    return scores