def score_pool(context):
    """Adapts uncertainty weighting based on progress and objective spread while incorporating novelty via distance to existing observations."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"] 
    campaign = context["campaign"]

    # Progress-aware exploitation vs exploration balance
    progress = campaign["progress"]
    
    # Dynamic weight: more exploit early, more explore later  
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * progress))
        
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Normalized uncertainty
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Combine exploitation and uncertainty with dynamic weight 
        score_exploit = w_exploit * mu_sum_norm  
        score_uncertainty = (1 - w_exploit) * sigma_norm
        
        base_score = score_exploit + score_uncertainty

        # Add novelty bonus: candidates farther from observed points get higher scores
        x_cand = cand["x"]
        
        if len(X_obs) > 0:
            distances = np.linalg.norm(x_cand - X_obs, axis=1)
            min_distance = np.min(distances)
            
            # Normalize by the range of features to make this scale-invariant  
            feature_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
            if not all(feature_range == 0):
                normalized_dist = min_distance / np.mean(feature_range[feature_range > 0])
                
                # Bonus for being far from existing observations (novelty reward)
                novelty_bonus = max(0, 1.5 - normalized_dist) 
                base_score += 0.2 * novelty_bonus
        
        scores.append(base_score)

    return scores