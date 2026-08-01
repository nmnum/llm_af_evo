def score_pool(context):
    """Exploitation with early uncertainty bonus: higher predicted means, boosted by uncertainty in early stages."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Early exploration bonus: add uncertainty scaled by remaining budget
        uncertainty_bonus = sum(gp[name]["std"] for name in names) * (1 - progress)
        scores.append(mu_sum + uncertainty_bonus)
    return scores