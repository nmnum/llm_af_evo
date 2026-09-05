def score_pool(context):
    """Blend hypervolume acquisition with uncertainty-aware coverage targeting and adaptive novelty penalty to balance exploration and exploitation."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = np.array(context["ref_point"])
    
    # Base scores from normalized acquisition values  
    acq_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Compute distances to Pareto front and reference point
    pf_distances, ref_distances = [], []
    for i, cand in enumerate(context["pool"]):
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Distance from candidate mean to nearest pareto point  
        if len(pf) > 0:
            dists_to_pf = [np.linalg.norm(gp_mean - pf_point, ord=2) 
                           for pf_point in pf]
            min_dist Pf = min(dists_to_pf)
        else:   
            min_dist_PF = np.inf
            
        # Distance to reference point (hypervolume expansion potential)
        ref_dist = np.linalg.norm(ref_point - gp_mean, ord=2)

        pf_distances.append(min_dist_PF)  
        ref_distances.append(ref_dist)

    # Normalize distances
    if len(pf) > 0:
        max_pf_distance = max(pf_distances)
        normalized_pf_dists = [d / max_pf_distance if d != np.inf else 1.0 
                               for d in pf_distances]
    else:    
        normalized_pf_dists = [1.] * len(context["pool"])

    # Normalize reference distances
    max_ref_dist = max(ref_distances)  
    normalized_ref_dists = [d / max_ref_dist if max_ref_dist > 0 else 0. 
                            for d in ref_distances]

    progress = context['campaign']['progress']
    
    # Early: emphasize uncertainty and coverage; late: favor exploitation
    exploit_weight = min(1., max(0., (2 * (1 - progress)) ** 3))
    explore_weight = 1.0 - exploit_weight
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
        
        # Uncertainty bonus using normalized stds
        sigma_sum_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names)
            
        uncertainty_bonus = explore_weight * (sigma_sum_normed / len(names))
                
        # Coverage term: how far candidate is from existing front  
        coverage_score = normalized_pf_dists[i]
        
        # Reference point distance as proxy of hypervolume expansion
        ref_point_score = 1.0 - normalized_ref_dists[i] 

        acq_value_norm = cand["acq_value_norm"]
            
        combined_coverage = (coverage_score + ref_point_score) / 2
        
        final_score = (
            acq_value_norm 
          + uncertainty_bonus * 3
          + 0.5 * explore_weight * combined_coverage  
          - 0.1 * exploit_weight * normalized_pf_dists[i]
        )
        
        scores.append(final_score)
    
    return scores