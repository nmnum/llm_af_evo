def score_pool(context):
    """Blend hypervolume improvement with a coverage-gap term: candidates near sparse front regions get bonus scores."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    Y_obs = context["Y_obs"] 
    k_nearest = 3
    
    # Compute distances from each candidate to the nearest points in pareto front or observations
    coverage_scores = []
    
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(pf) >= k_nearest:
            # Use Pareto front directly to find nearest neighbors 
            distances = [np.linalg.norm(pred_obj - pf_point, ord=2) for pf_point in pf]
        else:  
            # Not enough points on the pareto front; fall back to all observations
            distances = [np.linalg.norm(pred_obj - obs_point, ord=2) for obs_point in Y_obs]

        nearest_dists = sorted(distances)[:k_nearest] 
        mean_dist_to_front = np.mean(nearest_dists)
        
        coverage_scores.append(mean_dist_to_front)

    # Normalize the scores to be on a similar scale as acq_value_norm (which is [0,1])
    if len(coverage_scores) > 0:
        max_cov_score = float(np.max(coverage_scores))
        min_cov_score = float(np.min(coverage_scores)) 
        range_cov = max_cov_score - min_cov_score
        if abs(range_cov) < 1e-9:  
            normalized_coverage = np.zeros_like(coverage_scores)
        else:
            # Invert so that lower distance (i.e. more covered region near candidate) scores low.
            inverted_coverages = [max_cov_score - score for score in coverage_scores]
            normalized_coverage = np.array(inverted_coverages)/range_cov
    else:
        normalized_coverage = []

    
    blend_factor=0.1  # Small weight on the novel front-region term 
    final_scores = []
    for i, cand in enumerate(context["pool"]):
        base_acq_score = float(cand['acq_value_norm'])
        
        if len(normalized_coverage) >i:
            coverage_bonus = normalized_coverage[i]
        else:  
            # Should not happen due to same loop but just be safe
            coverage_bonus=0.0

        total_score =  base_acq_score + blend_factor * coverage_bonus 
        final_scores.append(total_score)

    return final_scores