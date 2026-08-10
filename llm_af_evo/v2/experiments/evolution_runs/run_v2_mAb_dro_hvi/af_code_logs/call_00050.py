def score_pool(context):
    """
    Exploitation with probabilistic pareto dominance: rank candidates by expected hypervolume improvement,
    weighted by how likely each is to be Pareto optimal based on GP posterior samples.
    """
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate probability of being Pareto optimal via sampling
        n_samples = 100
        dominates_count = 0
        
        # Sample from this candidate's GP posteriors 
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # For each sample, check if it dominates any current Pareto point
        cand_point = np.array([gp[name]["mean"] for name in names])
        is_pareto_sampled = True  # Start by assuming this candidate's mean is potentially pareto
        
        for pf_point in context["pareto_front"]:
            # If sample strictly improves on a front point, it doesn't dominate
            if np.all(cand_point >= pf_point) and not np.array_equal(cand_point, pf_point):
                dominates_count += 1
                
        prob_pareto = max(0.05, min(0.95, (n_samples - dominates_count)/ n_samples))
        
        # Use hypervolume improvement estimate as exploitation score
        cand_hv_improvement = np.prod(np.maximum(ref_point - cand_point, 0)) 
        
        scores.append(cand_hv_improvement * prob_pareto)
    
    return scores