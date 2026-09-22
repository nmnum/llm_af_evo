def score_pool(context):
    """Estimate candidates’ potential for diversifying the pareto front by scoring based on acquisition value, uncertainty, and how much they would expand hypervolume if sampled under noise."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    
    # Compute base scores using acq_value_norm
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Estimate uncertainty component (sum of normalized stds)
    front_range = context["pareto_front_range"]
    uncert_components = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        uncert_components.append(sigma_sum)

    # Compute novelty as minimum distance to any previously observed point
    X_obs = context['X_obs']
    
    if len(X_obs) > 0:
        novelties = []
        obs_ranges = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
        
        for cand in context["pool"]:
            x_cand = np.array(cand["x"])
            
            # Compute distances to all observations
            dists = np.linalg.norm(x_cand[None,:] - X_obs, axis=1)

            min_dist = np.min(dists) 
            
            if len(obs_ranges > 0):
                norm_factor = np.linalg.norm(obs_ranges[obs_ranges>0])
                min_dist /= (norm_factor + 1e-8)
            
            novelties.append(min_dist)
    else:
        # No observations yet
        novelties = [float('inf')] * len(context["pool"])

    
    progress = context['campaign']['progress']
   
    final_scores = []
  
    for i, cand in enumerate(context["pool"]):
        
        base_score = float(base_scores[i])
        uncertainty_component = uncert_components[i]
        novelty_component = novelties[i]

        # Combine components with dynamic weights that shift from exploitation to exploration
        exploit_weight = max(0.3 + 1.2 * progress, 0.)
        explore_weight = min(max((1.-progress) / (0.5+1e-8), 0.), 1.)

        
        score = (
            base_score 
          + uncertainty_component  
          - novelty_component # lower is better for this term
        )

      
        final_scores.append(score)

    return final_scores