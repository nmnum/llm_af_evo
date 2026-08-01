def score_pool(context):
    """Score by predicted objective sum plus uncertainty bonus scaled by progress, normalized by front range, and adjusted for stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Dynamic weight: increases with progress, but is reduced if stagnating
        weight = (1.0 + 2.0 * progress) * (1.0 - 0.3 * min(stagnant_batches, 5))
        scores.append(mu_sum + weight * sigma_norm)
    return scores