def score_pool(context):
    """Exploitation with dynamic uncertainty bonus: score by predicted objective sum plus UCB-style uncertainty scaled by progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Scale uncertainty bonus by progress: less exploration early, more later
        ucb_weight = 2.0 * (1.0 - progress)
        scores.append(mu_sum + ucb_weight * sigma_norm)
    return scores