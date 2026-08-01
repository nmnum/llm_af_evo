def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and normalization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores