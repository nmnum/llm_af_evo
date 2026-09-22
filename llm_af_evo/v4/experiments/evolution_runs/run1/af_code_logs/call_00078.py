def score_pool(context):
    """Estimate hypervolume expansion potential by perturbing candidates' GP means and measuring how often they dominate."""
    import numpy as np
    
    # Use a small amount of jitter for stability in estimation, not too much that it becomes noisy.
    noise_scale = 0.1
    n_samples_per_candidate = 256

    scores = []
    
    ref_point = context["ref_point"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]

    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy versions of this candidate's mean
        means_samples = []
        stds = [gp_posterior[name]["std"] for name in names]
        
        np.random.seed(0)  # For reproducibility, but not necessarily needed if we're consistent across all candidates
        
        for _ in range(n_samples_per_candidate):
            sample_means = [
                gp_posterior[names[i]]["mean"]
                    + noise_scale * stds[i] * np.random.randn()
                      for i in range(len(names))
                  ]
            
            means_samples.append(sample_means)
        
        # Compute hypervolume improvement across all samples
        hv_improvements = []
    
        for sample_mean in means_samples:
            candidate_point = np.array(sample_mean)  # shape (2,)
                
            if not any(np.all(candidate_point >= ref_point)):
                continue

            dominates_any_current_front = False
            
            current_pf_points = context["pareto_front"]
            
            dominated_by_some_in_PF = [
                 all(current_pt >= candidate_point)
                     for current_pt in current_pf_points
               ]
            
            # If no point on the front is better than this sample, it might contribute to HV improvement.
            if not any(dominated_by_some_in_PF):
                hv_improvements.append(1.0)  # Placeholder contribution
                
        score = np.mean(hv_improvements) if len(hv_improvements)>0 else -np.inf
        scores.append(score)

    return [s for s in scores]