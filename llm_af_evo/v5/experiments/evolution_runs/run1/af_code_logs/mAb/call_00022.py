def score_pool(context):
    """Estimate probability of a candidate improving front hypervolume by sampling from noisy GP posterior and measuring dominated volume expansion."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use acquisition value as base score
    scores = []
    n_samples = 10
    
    for cand in context["pool"]:
        acq_value = cand["acq_value_norm"]

        # Sample noisy predictions from GP posterior 
        samples = []  
        for _ in range(n_samples):
            sample_y = [cand["gp_posterior"][name]["mean"] + np.random.normal() * cand["gp_posterior"][name]["std"]
                        for name in names]
            samples.append(sample_y)

        # Estimate hypervolume improvement by seeing how many new samples dominate ref_point
        dominated_count = 0  
        for y in samples:
            if all(y[i] > ref_point[i] for i in range(len(names))): 
                dominated_count += 1
                
        p_dominance = dominated_count / n_samples
        
        # Blend acquisition value with dominance probability (scale down to avoid overfitting)
        score = acq_value + 0.2 * p_dominance
        scores.append(score)

    return scores