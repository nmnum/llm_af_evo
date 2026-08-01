def score_pool(context):
    """Exploitation-plus-uncertainty with progress-aware weighting: higher progress = more exploitation, less uncertainty bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight exploitation vs uncertainty: early = more UCB, late = more exploitation
        weight_exploit = 0.3 + 0.7 * (1 - progress)  # Decrease uncertainty bonus as progress increases
        weight_uncertainty = 1.0 - weight_exploit
        scores.append(weight_exploit * mu_sum + weight_uncertainty * 2.0 * sigma_sum)
    return scores