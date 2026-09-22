def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware sigmoidal blending while scaling uncertainties with adaptive factors tied to observed front spread."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation signal as normalized mean sum 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty scaled by a factor that decays over progress and is modulated by objective range
        sigma_scaled_sum = 0.0
        for name in names:
            std_val, range_val = gp[name]["std"], front_range[name]
            # Progress-adaptive decay of uncertainty contribution  
            adaptive_factor = np.exp(-3 * progress)
            scaled_std = std_val * adaptive_factor / range_val 
            sigma_scaled_sum += scaled_std

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_scaled_sum
        scores.append(score)

    return scores