def score_pool(context):
    """Integrate acquisition value with a front density-aware uncertainty bonus and adaptive novelty penalty to guide exploration towards under-covered yet uncertain regions."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from all previously observed points
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        novelty_distances = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            dists_to_observed = [np.linalg.norm(x_cand - x_obs) for x_obs in X_obs]
            min_dist = min(dists_to_observed)
            novelty_distances.append(min_dist)

        # Normalize distances to [0, 1] scale
        if max(novelty_distances) > 0:
            novel_scores = np.array(novelty_distances) / max(novelty_distances)
        else:
            novel_scores = np.zeros_like(novelty_distances)
    else: 
        novel_scores = np.zeros(len(context["pool"]))

    
    # Progress-aware scaling of uncertainty bonus
    progress = context['campaign']['progress']
        
    scores = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior'] 
        
        mu_sum = sum(gp_posterior[name]["mean"] for name in names)
                
        # Uncertainty bonus scaled by campaign phase and inverse of front density
        sigma_normed = np.mean([gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names])
        
        # Estimate how sparse the current Pareto front is (lower means more sparsity)
        pf_size = len(context['pareto_front'])
        if pf_size > 1:
            # Use a simple measure of density: average distance between points
            distances = []
            for j, p1 in enumerate(context["pareto_front"]):
                for k, p2 in enumerate(context["pareto_front"][j+1:], start=j+1): 
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    if not np.isnan(dist):
                        distances.append(dist)
            front_density_score = 0.5 * (np.mean(distances) / max(1e-6, context["pareto_front_range"][names[0]])) if len(distances) > 0 else 0
        elif pf_size == 1:
            # If only one point in the PF assume low density 
            front_density_score = 1.0  
        else:   
            front_density_score = 0
        
        uncertainty_bonus = (2 - progress)**3 * sigma_normed / max(1e-6,front_density_score) 
        
        # Inverse novelty penalty — penalize candidates too close to previous observations and 
        novel_penalty = min(novel_scores[i], 0.5)
                
        combined_score = acq_scores[i] + uncertainty_bonus - novel_penalty
        
        scores.append(combined_score)

    return scores