def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo samples from each candidate's GP posteriors."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    scores = []
    
    # Sample objectives for each candidate
    n_samples = 100
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint posterior of all objectives 
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Simple diagonal covariance matrix (uncorrelated)  
        cov_matrix = np.diag(stds**2)
        
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples)

        # Compute hypervolume improvement estimate
        hv_improvement = 0.0
        
        for sample in samples:
            if all(sample > ref_point): 
                continue
            
            dominated_by_front = False  
            front_sampled = context["pareto_front"]
            
            # Check domination against current Pareto front points (simplified)
            for pf_pt in front_sampled:                
                if all(pf_pt >= sample) and any(pf_pt > sample):
                    dominated_by_front = True
                    break
                    
            if not dominated_by_front:
                hv_improvement += 1.0
                
        scores.append(hv_improvement / n_samples)
        
    return scores