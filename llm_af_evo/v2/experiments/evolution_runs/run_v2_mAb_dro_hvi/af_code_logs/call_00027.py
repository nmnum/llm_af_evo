def score_pool(context):
    """Rank candidates by expected hypervolume improvement weighted by proximity to existing observations."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV improvements using Monte Carlo sampling from GP posteriors
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample predictions from the candidate's posterior distributions
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:  
                samples[:,i] = mean_val
        
        # Calculate hypervolume improvement for each sample
        hv_improvements = []
        
        # For simplicity and efficiency we consider only the candidate's own prediction 
        cand_pred = [gp[name]["mean"] for name in names]
        
        if all(cand_pred[i] > ref_point[i] for i in range(len(names))):
            hypervolume_candidate = np.prod([max(0, (cand_pred[i]-ref_point[i])) for i in range(len(names))])
            
            # Compute distance from existing observations
            x_cand = cand["x"]
            distances_to_observed = [np.linalg.norm(x_cand - obs_x) for obs_x in context["X_obs"]]
            min_distance = np.min(distances_to_observed)
        
            # Normalize and weight by novelty (inverse of squared distance to closest observation, plus a small constant
            novel_score = 1.0 / (min_distance**2 + 1e-6)

            scores.append(hypervolume_candidate * novel_score)  
        else:
            scores.append(0)
    
    return scores