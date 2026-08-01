def score_pool(context):
    """Exploitation-weighted score with progressive uncertainty bonus based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Progressive weight: start with 4.09 exploitation, reduce uncertainty bonus as progress increases
        exploit_weight = 4.09
        ucb_bonus = (1.0 - progress) * sigma_norm  # Reduce uncertainty bonus as we get closer to end
        scores.append(exploit_weight * mu_sum + ucb_bonus)
    return scores