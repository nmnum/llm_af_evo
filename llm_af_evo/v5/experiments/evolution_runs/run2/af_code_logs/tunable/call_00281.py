def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable predictions."""
    import numpy as np

    n_samples = 20
    lam = 1.0
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples from the candidate's posterior per objective
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])

        # Compute hypervolume improvement for each sample, adjusted if dominated
        hv_improvements = []
        
        front = context["pareto_front"]
        
        for s in samples:
            is_dominated = False
            
            # Check domination by any point on the current Pareto front 
            if len(front) > 0:  
                for p in front:
                    dominates_or_equal = np.all(p >= s)
                    strictly_better = np.any(p > s)

                    if dominates_or_equal and strictly_better:
                        is_dominated = True
                        break

            # Calculate raw hypervolume improvement, discounting dominated samples 
            volume_to_ref_point = np.prod(np.maximum(s - ref_point, 0))
            
            adjusted_volume = volume_to_ref_point * (1.0 if not is_dominated else 0.1)
                
            hv_improvements.append(adjusted_volume)

        # Compute mean and std of the improvement values
        mean_imp = float(np.mean(hv_improvements)) 
        std_imp = float(np.std(hv_improvements))

        score = mean_imp - lam * std_imp

        scores.append(score)
    
    return scores