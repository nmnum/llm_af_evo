def score_pool(context):
    """Blend acquisition value with uncertainty-aware novelty and progress-adaptive exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Base scores from hypervolume improvement estimate
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Normalize uncertainties relative to front range  
    normalized_stds = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        std_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        normalized_stds.append(std_sum)

    # Compute novelty as distance to nearest observed point
    X_obs = context["X_obs"]
    X_pool = np.stack([cand['x'] for cand in context["pool"]])
    
    novelties = []
    if len(X_obs) > 0:
        distances_to_observed = cdist(X_pool, X_obs)
        min_distances = np.min(distances_to_observed, axis=1)
        # Invert so larger distance means higher novelty
        max_dist = np.max(min_distances)
        novelties = (max_dist - min_distances + 1e-8) / (max_dist + 1e-8) if max_dist > 0 else np.zeros_like(min_distances)
    else:
        # No observations yet, all equally novel  
        novelties = [1.0] * len(X_pool)

    progress = context["campaign"]["progress"]
    
    scores = []
    for i in range(len(context["pool"])):
        
        base_score = base_scores[i]
        uncertainty = normalized_stds[i]
        novelty = novelties[i]

        # Early: balance acquisition + uncertainty
        if progress < 0.3:
            score = (base_score * 2) + (uncertainty * 1)
            
        elif progress < 0.7:
            # Mid campaign, exploit more but still use some exploration 
            exploitation_weight = min(1., max(0.5, 2 - progress*2))
            uncertainty_weight = 3*(progress-0.3)/(0.4) if (progress > 0.3) else 0
            
            score = base_score * exploitation_weight + \
                    novelty * 0.8 + \
                    uncertainty * uncertainty_weight
                    
        # Late: heavily favor acquisition value and diversity  
        elif progress >= 0.7:
            
            exploration_factor = min(1., (progress - 0.5) / 0.3)
                
            score = base_score*2 + novelty*(exploration_factor)*4
            
        
        scores.append(score)

    return scores