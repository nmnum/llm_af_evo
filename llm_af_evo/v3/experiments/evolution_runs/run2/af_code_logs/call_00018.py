def score_pool(context):
    """Exploitation-weighted uncertainty with hypervolume-aware normalization and progress-adaptive blending."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]
    
    # Adaptive weight for exploitation vs exploration
    w_exploit = 1.0 - max(0.0, min(1.0, (2 * progress) ** 3))
    w_uncertain = 1.0 - w_exploit
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and normalized stds
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Normalize the mean score based on current front range to avoid bias towards objectives with larger scales 
        norm_mu = mu_sum / len(names)  # simple average of means
        scaled_sigma = sigma_norm
        
        # Blend exploitation and uncertainty scores, weighted by progress  
        score = w_exploit * norm_mu + w_uncertain * scaled_sigma

        scores.append(score)
    
    return scores