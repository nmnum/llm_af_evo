def score_pool(context):
    """Exploitation with uncertainty-aware ranking: sum of means plus a scaled std penalty to balance exploration and exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Scale the uncertainty term to balance exploitation (mu) and exploration (sigma).
        scores.append(mu_sum - 0.5 * sigma_norm)
    return scores