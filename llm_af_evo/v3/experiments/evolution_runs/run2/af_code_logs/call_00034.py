def score_pool(context):
    """Progressively shift from hypervolume-aware uncertainty scoring to novelty-driven exploration."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        # Predicted objective values
        mu_vals = np.array([gp[name]["mean"] for name in names])
        
        # Hypervolume improvement estimate (normalized by front range)
        ref_distances = ref_point - mu_vals  
        hv_improvement = np.prod(ref_distances / list(front_range.values()))
        
        # Uncertainty-normalized score
        sigma_norm = sum(gp[name]['std'] / front_range[name] for name in names) 
        
        # Blend based on progress: early=more exploitation, late=more exploration 
        exploit_weight = max(0.1, 1 - progress * 2)
        
        if hv_improvement > 0:
            score = (exploit_weight * mu_vals.mean() + \
                    (1-exploit_weight) * sigma_norm)  
        else: # If dominated or near front
            min_dist_to_observed = float(np.linalg.norm(X_obs - cand['x'], axis=1).min())
            
            if progress < 0.5:
                score = hv_improvement + np.log(1e-6+sigma_norm)
            elif progress >= 0.8 and min_dist_to_observed > 0.2: 
                 # Encourage diversity late in campaign
                novelty_score = (min_dist_to_observed - 0.5) / max(min_dist_to_observed, 1e-3)
                score = hv_improvement + sigma_norm * exploit_weight + \
                        novelty_score*(1-exploit_weight)*2  
            else:
                 # Standard uncertainty scoring for middle stages
                score = (exploit_weight*sigma_norm - 
                         np.log(1+np.exp(-hv_improvement)))  # Soft dominance penalty

        scores.append(score)
    return scores