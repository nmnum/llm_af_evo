def score_pool(context):
    """Score candidates by hypervolume improvement estimate plus novelty reward, using GP samples to simulate future front expansion."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use Monte Carlo sampling from each candidate's posterior to estimate HV impact
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives for this candidate multiple times
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # For each sample, compute how much HV would increase if we added it to the current front
        hv_improvements = []
        for s in samples:
            expanded_front = np.vstack([context["pareto_front"], s])
            # Compute hypervolume of new front (assuming ref_point is lower bound)
            try:
                hv_new = compute_hypervolume(expanded_front, ref_point) 
                hv_old = compute_hypervolume(context["pareto_front"], ref_point)
                hv_improvements.append(max(0.0, hv_new - hv_old))
            except Exception:  # Fallback if HV computation fails
                hv_improvements.append(0.)
        
        expected_hv_impact = np.mean(hv_improvements) 
        
        # Add novelty bonus based on distance to nearest observed point (in feature space)
        cand_x = cand["x"]
        distances_to_observed = [np.linalg.norm(cand_x - obs_x) for obs_x in context["X_obs"]]
        min_distance = min(distances_to_observed) if len(distances_to_observed) > 0 else float('inf')
        
        # Novelty score: inverse of distance (higher is better), or zero if no observed points
        novelty_bonus = max(1e-6, 1. / (min_distance + 1e-9))  
                
        scores.append(expected_hv_impact * 2000 + novelty_bonus)
        
    return scores

def compute_hypervolume(front, ref_point):
    """Compute the hypervolume of a Pareto front relative to reference point."""
    from scipy.spatial import ConvexHull
    if len(front) == 0:
        return 0.0
    
    # Simplified version: assume all points dominate each other in some sense.
    # For full generality, one would use the actual hypervolume calculation 
    front = np.array(front)
    
    prod_vol = 1.0
    for i in range(len(ref_point)):
        if len(front) > 0:
            vol_i = max(0., ref_point[i] - min(front[:,i]))
        else:  
            vol_i = ref_point[i]
        prod_vol *= vol_i
    
    return prod_vol