def score_pool(context):
    """Balances exploitation and uncertainty with progressive weighting adjusted by stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress, stagnant_batches = context["campaign"]["progress"], context["campaign"]["stagnant_batches"]

    # Dynamic blend: exploration-heavy early, exploit later; increase weight if stagnating
    w_exploit_base = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    w_exploit = w_exploit_base * (1.0 + stagnant_batches * 0.2)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)

        # Adaptive uncertainty with decay based on progress and stagnation  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress) - stagnant_batches * 0.05)
            / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled  
        scores.append(score)

    return scores