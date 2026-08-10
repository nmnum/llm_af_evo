def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo sampled GP posteriors for each candidate."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]

    # Use a fixed number of MC samples to estimate expected HV improvement
    n_samples = 100
    scores = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the joint posterior distribution (simplified: independent normals)
        sampled_objectives = []
        for _ in range(n_samples):
            sample = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            sampled_objectives.append(sample)

        # Compute hypervolume improvement estimate
        hv_improvements = []

        for obj_sample in sampled_objectives:
            # Check if this sample dominates the reference point (i.e., all objectives are better)
            dominated_by_front = False  # Simplified: assume no domination by front, only ref_point
            
            is_better_than_ref = np.all(np.array(obj_sample) > ref_point)

            hv_improvement = max(0.0,
                                 sum((obj - rp) for obj, rp in zip(obj_sample, ref_point)) if is_better_than_ref else 0.)
            
            # Normalize by the range of observed objectives to make score scale-invariant
            norm_factor = np.prod([front_range[name] for name in names])
            hv_improvement /= (norm_factor + 1e-8) 
                
            hv_improvements.append(hv_improvement)
        
        scores.append(np.mean(hv_improvements))

    return scores