def score_pool(context):
    """Estimates hypervolume improvement potential by sampling noisy predictions and measuring expansion."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of samples to estimate HV improvement, as in the baseline
    n_samples = 10
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample noisy predictions from GP posteriors (mean ± N(0,std))
        hv_improvements = []

        for _ in range(n_samples):
            sampled_objectives = [np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                                  for name in names]
            
            # Compute hypervolume contribution of this sample
            if all(sampled_objectives[i] > ref_point[i] for i in range(len(names))):
                hv_improvements.append(1.0)  # Dominated by reference point, full HV gain possible  
            else:
                # Simplified: compute normalized distance to front (not exact hypervolume)
                dist_to_front = min(
                    sum((sampled_objectives[i] - ref_point[i])**2 for i in range(len(names)))**0.5
                        if sampled_objectives[i] <= ref_point[i]
                            else 1e6 
                                for _pf in context["pareto_front"]
                                    for (i, _) in enumerate(ref_point)
                                        )
                hv_improvements.append( max(dist_to_front - np.linalg.norm(front_range),0) )

        # Score is the expected HV improvement
        scores.append(np.mean(hv_improvements))
    
    return scores