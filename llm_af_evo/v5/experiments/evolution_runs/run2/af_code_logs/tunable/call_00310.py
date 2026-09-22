def score_pool(context):
    """Integrate hypervolume acquisition with dynamic uncertainty scaling and inverse front density to prioritize under-explored yet promising regions."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    campaign = context["campaign"]

    # Compute inverse front density score: candidates near sparse regions get higher scores
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        densities = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Use k-nearest neighbors to estimate local density (k=5 or all points if fewer)  
            k = min(5, len(X_obs))
            nearest_dists = sorted(dists)[:k]
            avg_dist_to_neighbors = np.mean(nearest_dists)

            # Inverse of average distance gives a notion of sparsity
            inv_density_score = 1.0 / max(avg_dist_to_neighbors, 1e-8)
            
            densities.append(inv_density_score) 
            
    else:
        densities = [1.] * len(context["pool"])

    
    front_range = context["pareto_front_range"]
        
    # Compute uncertainty bonus with progress-aware scaling
    ucb_weight = np.exp(-campaign["progress"]*3) 
    
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        # Scale uncertainty bonus based on campaign progress
        scaled_uncertainty_bonus = ucb_weight * sigma_sum
        
        unc_scores.append(scaled_uncertainty_bonus)

    final_scores = acq_values + np.array(densities) - 0.5*np.array(unc_scores)
    
    return list(final_scores)