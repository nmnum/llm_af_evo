def score_pool(context):
    """Score candidates by hypervolume improvement estimate adjusted for novelty distance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV improvement using Monte Carlo sampling from GP posteriors
    n_samples = 100
    hv_improvements = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives from the candidate's posterior distribution
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Compute hypervolume improvement estimate
        hv_improvement = 0.0
        
        for sample_obj in samples:
            if all(sample_obj[i] >= ref_point[i] for i in range(len(names))):
                continue
            
            dominated_by_front = False
            for front_pt in context["pareto_front"]:
                # Check domination: is the sampled point worse than or equal to a Pareto point?
                dominates = True
                for j, val in enumerate(sample_obj):
                    if val < front_pt[j]:
                        dominates = False
                        break
                
                if dominates:
                    dominated_by_front = True
                    break
            
            if not dominated_by_front:  # Sampled objective is potentially improving HV
                hv_improvement += np.prod(ref_point - sample_obj)
        
        hv_impr_mean = hv_improvement / n_samples
        
        # Add novelty bonus (distance to nearest observed point in feature space)  
        cand_x = cand["x"]
        min_dist_to_observed = float('inf')
        for obs_x in context["X_obs"]:
            dist_sq = np.sum((cand_x - obs_x)**2)
            if dist_sq < min_dist_to_observed:
                min_dist_to_observed = dist_sq
        
        novelty_bonus = 1.0 / (min_dist_to_observed + 1e-8) # avoid division by zero

        hv_impr_mean += 5 * novelty_bonus
        hv_improvements.append(hv_impr_mean)
        
    return hv_improvements