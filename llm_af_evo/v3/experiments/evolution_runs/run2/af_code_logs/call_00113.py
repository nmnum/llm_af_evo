def score_pool(context):
    """Progress-aware hypervolume signal with dynamic exploitation-uncertainty balance and normalized uncertainty scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal from mean objectives 
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with dynamic scaling based on progress and objective ranges
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        
        # Add a novelty bonus based on distance to nearest observed point  
        x_cand = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances) 
            # Normalize by the range of features
            feature_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
            normalized_dist = min_distance / (np.mean(feature_range) + 1e-8)
            score += 0.5 * (1.0 - progress) * normalized_dist
        
        scores.append(score)

    return scores