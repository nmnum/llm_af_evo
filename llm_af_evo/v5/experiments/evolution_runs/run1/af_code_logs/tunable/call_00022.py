def score_pool(context):
    """Estimate candidates’ potential for improving hypervolume by sampling from noisy posterior means."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample noisy means (perturb each objective's mean by its std)
        samples = 10
        noisy_means = np.random.normal(
            [gp[name]["mean"] for name in names],
            [gp[name]["std"] for name in names],
            size=(samples, len(names))
        )

        # Compute hypervolume contribution of each sample (relative to ref point)
        hv_contributions = []
        for mean_sample in noisy_means:
            if np.all(mean_sample >= ref_point):  # Fully dominates reference
                contrib = float('inf') 
            else:  
                # Simple volume calculation assuming no overlap with front, only relative improvement
                vol_diffs = (ref_point - mean_sample) / [front_range[name] for name in names]
                if np.any(vol_diffs < 0): continue   # Invalid contribution
            
                contrib = np.prod(np.maximum(1e-9, vol_diffs)) 
            hv_contributions.append(contrib)
        
        score = sum(hv_contributions)/len(hv_contributions) if hv_contributions else -float('inf')
        scores.append(score)

    return scores