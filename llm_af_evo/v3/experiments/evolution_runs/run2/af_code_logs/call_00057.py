def score_pool(context):
    """Estimates hypervolume expansion potential by sampling posterior means and scoring based on how much a candidate could improve the current front."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Use progress to adjust exploration vs exploitation
    p = 1.0 - max(0.0, min(1.0, (2 * context["campaign"]["progress"]) ** 3))
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample predictions from the GP posteriors
        samples_per_objective = [np.random.normal(gp_posterior[name]["mean"], 
                                                  max(1e-6, gp_posterior[name]["std"]), 50) for name in names]
        
        sample_means = np.array(samples_per_objective).T
        
        # Estimate hypervolume improvement by comparing to current front
        hv_improvement_scores = []
        for mean_sample in sample_means:
            if all(mean_sample >= ref_point):
                continue
            
            dominated_by_front = False 
            for point_in_pf in context["pareto_front"]:
                if np.all(point_in_pf <= mean_sample) and not np.array_equal(point_in_pf, mean_sample):  
                    dominated_by_front = True
                    break
                    
            # If the sample is better than current front or on a new frontier (not yet dominated)
            hv_improvement_scores.append(1.0 - max(np.min((ref_point-mean_sample)/(ref_point-context["pareto_front_range"]["f2"])), 0))
        
        if not hv_improvement_scores:
            score = np.mean([gp_posterior[name]["mean"] for name in names]) 
        else:  
            # Weighted blend of mean prediction and hypervolume signal
            mu_score = sum(gp_posterior[name]["mean"] for name in names)
            hvi_score = max(hv_improvement_scores) if hv_improvement_scores else 0.0
            
            score = p * (mu_score / len(names)) + (1 - p) * hvi_score
        
        scores.append(score)

    return scores