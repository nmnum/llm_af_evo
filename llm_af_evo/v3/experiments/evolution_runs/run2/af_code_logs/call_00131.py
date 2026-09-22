def score_pool(context):
    """Balances exploitation and exploration using adaptive weights derived from the hypervolume difference between predicted candidate front contribution and current reference point."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective means and stds  
        mu_vals = [gp[name]["mean"] for name in names]
        sigma_vals = [gp[name]["std"] for name in names]

        # Hypervolume contribution estimate using reference point
        hv_contribution = np.prod(np.maximum(ref_point - np.array(mu_vals), 0.0))
        
        # Normalize uncertainty by the range of objectives 
        normalized_sigma_sum = sum(sigma / (ref_point[i] - gp[names[i]]["mean"]) for i, sigma in enumerate(sigma_vals))

        score = w_exploit * hv_contribution + (1.0 - w_exploit) * np.sqrt(normalized_sigma_sum)
        
        scores.append(score)

    return scores