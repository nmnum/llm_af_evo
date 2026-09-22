def score_pool(context):
    """Resample posterior means with noise and evaluate hypervolume expansion potential."""
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"] 
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    n_samples = 10
    noise_level = 0.05
    
    # Estimate hypervolume improvement by resampling candidate's posterior 
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Generate noisy samples from GP posteriors (mean +/- std * N(0,1))
        f_samples = []
        for _ in range(n_samples):
            sample_means = [np.random.normal(gp[name]["mean"], 
                                             max(1e-6, noise_level*front_range[name]))  
                            for name in names]
            # Ensure we're sampling from the right direction (already flipped to maximize)
            f_samples.append(sample_means)

        hv_improvements = []
        
        # For each sample of objectives
        for fsample in f_samples:
            
            # Check if this candidate would dominate any existing points or extend hypervolume 
            is_dominant = True
            
            # Use a small epsilon to avoid numerical issues with equality checks  
            eps = 1e-6 
            
            hv_improvement = None
          
            try:    
                # Compute HV of the new point vs. current front (with ref_point)
                
                test_front = np.vstack([Y_obs, fsample])
            
                from scipy.spatial import ConvexHull
                
                if len(test_front) < 2:
                    hv_improvement = float('inf')
                    
                else: 
                    # Simple hypervolume calculation using reference point
                    ref_point_array = ref_point
                    
                    def volume_of_box(p1, p2):
                        return np.prod(np.maximum(0.0, (p1 - p2)))
                        
                    vol_ref = 1.
                    for i in range(len(ref_point)):
                        if abs(fsample[i] - ref_point[i]) > eps:
                            # This is a very rough estimation of hypervolume
                            hv_improvement = volume_of_box(np.array([fsample[0], fsample[1]]), 
                                                           np.array([ref_point_array[0],
                                                                     ref_point_array[1]]) )
                        else:  
                            hv_improvement = 0.0
                        
            except Exception as e:
                # fallback in case of numerical issues
                hv_improvement = -np.inf
                
            
            if not (isinstance(hv_improvement, float) and np.isfinite(hv_improvement)):
                 hv_improvements.append(1e-8)
                
        avg_hv_imp = sum(hv_improvements)/len(hv_improvements) 
          
        # Add novelty term to avoid duplicate selections
        min_dist_to_observed = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        
        score = (avg_hv_imp + 0.5 * np.exp(-3*min_dist_to_observed)) / max(1e-6, sum(gp[name]["std"] for name in names))
            
        scores.append(score)
    
    return scores