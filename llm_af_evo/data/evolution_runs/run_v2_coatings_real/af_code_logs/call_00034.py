def score_pool(context):
    """Exploitation with progress-adaptive uncertainty bonus: UCB-style score that shifts from exploration early to exploitation later."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adapt the balance between exploitation and exploration based on progress
        weight_exploitation = 1.0 - progress
        weight_exploration = progress
        scores.append(weight_exploitation * mu_sum + weight_exploration * sigma_sum)
    return scores