def score_pool(context):
    """Balances uncertainty-sensitive exploitation with progressive hypervolume optimism using dynamic reward shaping."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Early exploration via UCB, tapering off as we approach the end
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.6)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Reward candidates that are both promising (high mean) AND uncertain
        ucb_like_score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * np.sqrt(sigma_norm_sum)
        
        # Adjust based on how much the candidate's predicted objectives exceed current front bounds,
        # incentivizing expansion of dominated hypervolume in a progressive way.
        hv_improvement_signal = 0
        for name in names:
            pred_mean = gp[name]["mean"]
            ref_val = context["ref_point_by_name"][name]
            
            if pred_mean > ref_val:  
                hv_improvement_signal += (pred_mean - ref_val) / front_range[name]

        # Combine exploitation and hypervolume signals with a dynamic blend
        final_score = ucb_like_score + 0.5 * np.tanh(2.0 * progress) * hv_improvement_signal
        
        scores.append(final_score)
    
    return scores