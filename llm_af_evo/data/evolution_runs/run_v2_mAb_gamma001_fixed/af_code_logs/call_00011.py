def score_pool(context):
    """Exploitation with uncertainty-based diversity: sum of means plus bonus for prediction variance, scaled by progress."""
    names = context["objective_names"]
    scores = []
    progress = context["campaign"]["progress"]
    
    # Early on, favour exploration; later, exploit more.
    exploitation_weight = 0.3 + 0.7 * (1 - progress)
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)

        # Scale the uncertainty bonus by how far through the campaign we are
        score = exploitation_weight * mu_sum + (1 - exploitation_weight) * sigma_sum
        
        scores.append(score)
        
    return scores