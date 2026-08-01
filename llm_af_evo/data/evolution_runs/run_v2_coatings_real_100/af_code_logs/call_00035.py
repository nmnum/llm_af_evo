def score_pool(context):
    """Blend predicted objective sum and uncertainty with a dynamic weight that shifts from exploration to exploitation based on progress and stagnation."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with more uncertainty, reduce as progress increases
        # Increase weight if stagnant to encourage diversity
        w_exploit = 0.2 + 0.8 * (1 - progress) * (1 + 0.5 * min(stagnant_batches, 3))
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores