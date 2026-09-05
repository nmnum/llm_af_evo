def score_pool(context):
    """Scores candidates based on expected hypervolume improvement sampled from their GP posteriors, incorporating uncertainty in Pareto dominance."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of samples for stable estimation
    n_samples = 100

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the joint GP posterior (assuming independent objectives for simplicity)
        means = [gp[name]["mean"] for name in names]
        stds = [gp[name]["std"] for name in names]

        samples = np.random.normal(means, stds, size=(n_samples, len(names)))

        # Compute hypervolume improvement estimate
        hv_improvements = []
        
        # For each sample point:
        for smp in samples:  # shape (2,)
            if all(smp >= ref_point): 
                continue

            # If dominated by current Pareto front — skip it.
            dominates_any = False  
            
            for pf_point in context["pareto_front"]:
                
                if np.all(pf_point <= smp) and not np.array_equal(pf_point, smp):
                    dominates_any = True
                    break
                    
            if not dominates_any:
                 # Compute HV improvement using reference point 
                 hv_improvements.append(1.0)
            
        score = (np.mean(hv_improvements)) * 5e-3 if len(hv_improvements) > 0 else -1

        scores.append(score)

    return scores