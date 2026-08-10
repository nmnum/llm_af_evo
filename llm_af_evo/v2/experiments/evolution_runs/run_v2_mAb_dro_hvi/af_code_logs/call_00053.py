def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo-sampled GP posteriors for each candidate."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use a fixed number of MC samples to estimate HV contribution
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the joint GP posterior across all objectives for this candidate
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Simple diagonal covariance matrix (uncorrelated) — good enough approximation here 
        cov_matrix = np.diag(stds**2)
        
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples)

        # Compute hypervolume contribution of this candidate's sampled points
        hv_contributions = []
        for sample in samples:
            if all(sample >= ref_point):  # dominated by reference point - no HV gain  
                continue
            
            # Hypervolume is computed relative to the current reference (i.e., how much better than it)
            hyp_vol = np.prod(ref_point - np.minimum(sample, ref_point))
            
            hv_contributions.append(hyp_vol)

        if not hv_contributions:
            scores.append(0.0)  # No contribution
        else:
            score = sum(hv_contributions)/len(hv_contributions) 
            scores.append(score)
    
    return scores