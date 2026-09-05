def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware mean normalization combined with a novel entropy-based diversity signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal: mean of objectives scaled by front range
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with adaptive scaling based on progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        
        # Add a novel diversity bonus based on entropy of predicted means across objectives 
        mean_values = [gp[name]["mean"] for name in names]  
        if len(mean_values) > 1:
            p = np.array(mean_values)
            p /= (np.sum(p) + 1e-8)   # Normalize to prob distribution
            H = -np.sum(p * np.log(p + 1e-8))    # Shannon entropy 
            score += 0.5 * (H / np.log(len(names)))  # Normalized between [0, 1]

        scores.append(score)
    
    return scores