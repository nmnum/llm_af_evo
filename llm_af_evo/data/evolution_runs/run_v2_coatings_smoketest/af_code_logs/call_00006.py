def score_pool(context):
    """Approximate expected hypervolume improvement using Monte Carlo sampling from Gaussian posteriors."""
    import numpy as np
    
    n_samples = 100
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Draw samples from each objective's posterior
        samples = []
        for _ in range(n_samples):
            sample = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            samples.append(sample)
        samples = np.array(samples)
        
        # Compute hypervolume improvement for each sample
        hv_improvements = []
        for sample in samples:
            # Check if sample is dominated by current Pareto front
            is_dominated = False
            for front_point in context["pareto_front"]:
                if all(sample[i] <= front_point[i] for i in range(len(names))):
                    is_dominated = True
                    break
            if not is_dominated:
                # Compute hypervolume contribution of this sample
                # HV = product of distances from ref_point to sample in each dimension
                hv_contribution = np.prod(np.maximum(ref_point - sample, 0))
                hv_improvements.append(hv_contribution)
        
        # Average HV improvement across all samples
        if hv_improvements:
            scores.append(np.mean(hv_improvements))
        else:
            scores.append(0.0)
    
    return scores