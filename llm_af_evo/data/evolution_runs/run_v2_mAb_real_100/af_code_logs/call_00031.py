def score_pool(context):
    """Exploitation with dynamic weighting and uncertainty bonus, adapted from parent A and B."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with 4.085 (from B) and reduce uncertainty bonus as progress increases
        w_exploit = 4.085
        w_uncertainty = max(0.0, 2.0 * (1.0 - progress))  # Decrease uncertainty bonus over time
        scores.append(w_exploit * mu_sum + w_uncertainty * sigma_sum)
    return scores