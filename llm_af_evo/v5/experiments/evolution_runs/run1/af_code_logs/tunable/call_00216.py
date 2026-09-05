def score_pool(context):
    """Blend hypervolume improvement with a coverage-gap term: reward candidates near sparse front regions."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small for k=3 nearest neighbors
    use_y_obs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_means = [cand["gp_posterior"][name]["mean"] for name in names]
        
        # Compute mean distance to k-nearest front points (or Y_obs if insufficient)
        distances = []
        ref_points = pf if not use_y_obs else context["Y_obs"]
        
        from scipy.spatial.distance import cdist
        cand_matrix = np.array([gp_means])
        ref_matrix = ref_points[:, :len(names)]  # Ensure same number of objectives
        
        dists = cdist(cand_matrix, ref_matrix)[0]
        k_nearest_dists = sorted(dists)[:3 if not use_y_obs else min(3, len(ref_points))]
        
        mean_dist_to_front = np.mean(k_nearest_dists) if k_nearest_dists else 1.0
        
        # Normalize by front range
        frange = [context["pareto_front_range"][name] for name in names]
        normalized_mean_dist = mean_dist_to_front / (np.array(frange).mean() + 1e-8)
        
        acq_norm = cand["acq_value_norm"]
        coverage_gap_score = max(0.0, 1 - normalized_mean_dist) * 0.2
        scores.append(acq_norm + coverage_gap_score)

    return scores