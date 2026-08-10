def score_pool(context):
    """Score candidates by hypervolume improvement estimate adjusted for novelty distance."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV contribution per candidate using Monte Carlo sampling from GP posteriors
    n_samples = 100
    hv_scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objectives from the posterior distributions of this candidate
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Compute hypervolume contribution of this candidate's sampled points
        hv_contributions = []
        for sample in samples:
            if all(sample >= ref_point):  # Dominated by reference point (no HV gain)
                continue
            
            dominated_by_front = False  
            front_points = context["pareto_front"]
            
            # Check dominance against current Pareto frontier 
            for pf_point in front_points:   
                if all(pf_point <= sample) and any(pf_point < sample):
                    dominated_by_front = True
                    break
                    
            if not dominated_by_front:
                hv_contributions.append(1.0)
                
        avg_hv_contribution = np.mean(hv_contributions) if len(hv_contributions) > 0 else 0.
        
        # Add novelty bonus based on distance to nearest observed point  
        x_cand = cand["x"]
        distances_to_observed = [np.linalg.norm(x_cand - obs_x, ord=2) for obs_x in context['X_obs']]
        min_distance = np.min(distances_to_observed)
        
        # Normalize and scale novelty bonus
        novel_bonus = 1.0 / (min_distance + 1e-8)

        hv_scores.append(avg_hv_contribution * novel_bonus)
    
    return hv_scores