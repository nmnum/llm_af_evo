def score_pool(context):
    """Exploitation with adaptive uncertainty weighting: blend between fixed and progress-driven UCB based on stagnation."""
    names = context["objective_names"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Base UCB weight: start low, increase with progress
    base_weight = 0.5 + 0.5 * progress
    
    # Increase weight if we're stagnating to encourage exploration
    if stagnant_batches > 0:
        # Linearly scale up the weight as stagnation increases (up to a cap)
        weight = min(base_weight + 0.2 * stagnant_batches, 2.0)
    else:
        weight = base_weight
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + weight * sigma_sum)
    return scores