def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate posteriors against observed history to better estimate dominance under noise."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement by checking how often a candidate's sample would improve the current HV
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute hypothetical HV with this new point added 
            hyp_points = np.vstack((context["pareto_front"], cand_obj))
            hv_improvement = _hypervolume_contribution(hyp_points, ref_point)
            hv_improvements.append(hv_improvement)

        score = 1.0 * sum(hv_improvements) / n_samples
        scores.append(score)
    
    return scores

def _hypervolume_contribution(points, reference):
    """Compute the hypervolume contribution of adding a new point to an existing set."""
    if len(points) < 2:
        return float('inf')
        
    # Sort points by objective values (ascending for each axis), then compute HV
    sorted_points = np.array(sorted(points.tolist(), key=lambda x: (-x[0], -x[1])))
    
    hv_total = _hypervolume_simple(sorted_points, reference)
    if len(sorted_points) > 2:
        # Remove last point and recompute to estimate contribution of that one.
        reduced_set = sorted_points[:-1]
        hv_reduced = _hypervolume_simple(reduced_set, reference)
        
        return max(0.0, hv_total - hv_reduced)

    else: 
        return 0.0

def _hypervolume_simple(points_sorted_by_f2_then_f1_descending, ref):
    """Simple hypervolume calculation assuming sorted by f2 desc then f1 descending."""
    
    # This is a basic approximation using the reference point
    hv = float(0)
    for i in range(len(points_sorted_by_f2_then_f1_descending)):
        p_i = points_sorted_by_f2_then_f1_descending[i]
        
        if len(p_i) < 2:
            continue
            
        # Determine area bounded by this and next point (if exists), to the reference
        x_low, y_high = ref[0], ref[1] 
        try:  
            p_next_y_sorted = points_sorted_by_f2_then_f1_descending[i+1]
            
            if len(p_next_y_sorted) >= 2:
                # For each point (i), find the area bounded between it and next in sequence
                x_low, y_high = max(ref[0],p_i[0]), min(y_high,p_i[1])
        except IndexError:  
             pass
            
        
        hv += abs(p_i[0] - ref[0]) * abs(min(ref[1]-ref[1]+.5*abs((y_high-ref[1])), p_i[1]))
    
    return max(0.,hv)