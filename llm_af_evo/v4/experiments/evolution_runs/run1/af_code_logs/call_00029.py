def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and novelty penalty."""
    scores = []
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute average hypervolume improvement estimate from the pool
    hv_improvements = [cand['acq_value_norm'] for cand in context['pool']]
    avg_hv_imp = sum(hv_improvements) / len(hv_improvements)
        
    progress_factor = 1.0 - min(context["campaign"]["progress"], 1.0)

    # Compute novelty scores (inverse distance to nearest observed point)
    x_observed = context['X_obs']
    
    if not x_observed.size:
        novelties = [1.] * len(context['pool'])
    else: 
        cand_x_list = np.array([cand["x"] for cand in context['pool']])
        
        # Compute pairwise distances
        dists_sq = []
        for i, candidate_x in enumerate(cand_x_list):
            diff = x_observed - candidate_x  
            squared_distances = np.sum(diff**2,axis=1)
            min_dist_squared = np.min(squared_distances) 
            if min_dist_squared == 0:
                novelties.append(0.)
            else:    
                # Invert to get novelty (higher is better), avoid division by zero
                novel_score = 1. / np.sqrt(min_dist_squared + 1e-8)
                novelties.append(novel_score)

    for i, cand in enumerate(context['pool']):
        acq_value_normed = cand["acq_value_norm"]
        
        # UCB-style uncertainty bonus with progress sensitivity
        ucb_bonus = (0.5 * progress_factor) * sum(cand["gp_posterior"][name]["std"] 
                                                  for name in names)
            
        score = 1.2* acq_value_normed + ucb_bonus
        
        # Scale novelty by the inverse of average HV improvement to balance exploration
        scaled_novelty_penalty = novelties[i] / (avg_hv_imp if avg_hv_improvements > 0 else 1.)
        
        score -= max(0.3 *scaled_novelty_penalty, 0.) 
                
    return scores