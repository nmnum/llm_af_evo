def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate objectives under noise, weighted by distance to existing observations."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point if added to the current front 
            temp_front = np.vstack((context["pareto_front"], cand_obj))
            
            # Calculate hypervolume using reference point
            hv_contrib = 0.0
            
            for pt in temp_front:
                dominates_ref = (pt >= ref_point).all()
                
                if not dominates_ref:   # Only consider points that don't dominate the reference  
                    continue
                
                vol = np.prod(ref_point - pt)
                    
                # Adjust volume by a factor related to distance from existing observations
                dist_to_observed = min(np.linalg.norm(cand_obj - obs) for obs in X_obs) if len(X_obs) > 0 else float('inf')
                
                hv_contrib += (vol / max(1.0, dist_to_observed)) 
            
            hv_improvements.append(hv_contrib)
        
        score = np.mean(hv_improvements)

        scores.append(score)
    
    return scores