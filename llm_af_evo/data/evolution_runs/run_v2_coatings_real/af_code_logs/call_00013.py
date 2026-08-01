def score_pool(context):
    """Score by predicted objective sum plus uncertainty bonus scaled by progress, normalized by front range, with dynamic weighting."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Dynamic weight: start with 1.0, increase to 3.0 as progress increases
        weight = 1.0 + 2.0 * progress
        scores.append(mu_sum + weight * sigma_norm)
    return scores