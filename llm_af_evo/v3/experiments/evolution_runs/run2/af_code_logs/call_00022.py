def score_pool(context):
    """Exploitation-weighted uncertainty with hypervolume-aware normalization, progress-adaptive blending, and novelty bonus."""
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
        ucb_score = w_exploit * norm_mu + w_uncertain * scaled_sigma

        # Novelty term: inverse of distance to nearest observed point
        x_cand = cand["x"]
        if len(X_obs) == 0:
            novelty = 1e6  # No observations, so very novel
        else:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            novelty = 1. / (min_distance + 1e-9) if min_distance > 0 else 1e6

        scores.append(ucb_score + 0.5 * novelty)
    
    return scores