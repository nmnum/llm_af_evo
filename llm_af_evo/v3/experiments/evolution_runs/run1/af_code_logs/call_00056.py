def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to estimate hypervolume contribution under uncertainty."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate HV improvement by sampling from the candidate's own GP posterior
        n_samples = 100
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            # Sample from normal distribution (already oriented to be higher is better)
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume contribution for each sample
        hv_contributions = []
        
        for i in range(n_samples):
            candidate_point = samples[i]
            
            # If point dominates the reference or any existing front points,
            # it contributes to HV (but we need a proper dominance check)
            if np.all(candidate_point >= ref_point) and not np.any(np.all(candidate_point <= context["pareto_front"], axis=1)):
                hv_contrib = 0.5 * max(0, candidate_point[0] - ref_point[0]) ** 2
            else:
                # Use a simplified proxy: hypervolume improvement is roughly the volume 
                # of region dominated by this point (or if inside front area)
               hv_val = np.prod(np.maximum(ref_point - candidate_point, 0))
                
            hv_contributions.append(hv_val)

        scores.append(sum(hv_contributions) / n_samples) 

    return scores