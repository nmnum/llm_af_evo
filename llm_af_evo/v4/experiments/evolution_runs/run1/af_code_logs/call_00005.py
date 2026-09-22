def score_pool(context):
    """Estimate Pareto front uncertainty via bootstrap resampling, then compute average hypervolume improvement of each candidate across these fronts."""
    import numpy as np
    
    # Bootstrap parameters
    n_bootstrap = 20
    Y_obs = context["Y_obs"]
    
    # Reference point for HV computation (same order as objectives)
    ref_point = context["ref_point"]

    def dominates(p, q):
        """Check if p weakly dominates q strictly in at least one objective."""
        return np.all(q >= p) and np.any(q > p)

    def is_non_dominated(points):
        """Vectorized check for non-dominated points using correct dominance logic.""" 
        n = len(points)
        dom_matrix = np.zeros((n, n), dtype=bool)
        
        # Build domination matrix
        for i in range(n):
            for j in range(n):  
                if dominates(points[i], points[j]):
                    dom_matrix[i,j] = True
        
        return ~np.any(dom_matrix, axis=1)  # Points not dominated by any other

    def hypervolume(front, ref_point):
        """Compute total HV of front relative to reference point."""
        if len(front) == 0:
            return 0.0
        # Simple implementation for two objectives (can be extended)
        hv = np.prod(ref_point - np.max(front, axis=0))
        return max(0., hv)

    scores = []
    
    for cand in context["pool"]:
        
        gp_posterior = cand["gp_posterior"]
        pred_mean = [gp_posterior[name]["mean"] for name in context["objective_names"]]
            
        # Bootstrap resampling
        improvements = []

        for _ in range(n_bootstrap):
            sampled_indices = np.random.choice(len(Y_obs), size=len(Y_obs), replace=True)
            sample_Y = Y_obs[sampled_indices]
        
            # Get non-dominated points from this bootstrap front 
            is_ndom = is_non_dominated(sample_Y)  
            boot_front = sample_Y[is_ndom]

            if len(boot_front) == 0:
                hv_before, hv_after = 0., 0.
                
            else:    
                hv_before = hypervolume(boot_front, ref_point)
            
                # Add candidate to front and compute new HV
                extended_front = np.vstack([boot_front, pred_mean])
                is_ndom_cand = is_non_dominated(extended_front) 
                cand_front = extended_front[is_ndom_cand]
                
                hv_after = hypervolume(cand_front, ref_point)
            
            improvements.append(hv_after - hv_before)

        # Average improvement across bootstrap draws
        scores.append(np.mean(improvements))

    return scores