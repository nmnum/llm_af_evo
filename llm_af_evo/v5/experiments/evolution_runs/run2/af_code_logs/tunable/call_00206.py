def score_pool(context):
    """Score candidates by blending acquisition value with a dynamic uncertainty-weighted novelty term that adapts based on front sparsity and campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute per-candidate distances to nearest observed point (novelty)
    novelty_scores = []
    X_obs = context['X_obs']
    if len(X_obs) > 0:
        for cand in context["pool"]:
            x_cand = np.array(cand["x"])
            dists = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists)
            novelty_scores.append(min_dist)
    else:
        # If no observations yet, assume all candidates are equally novel
        novelty_scores = [0.0] * len(context["pool"])
    
    front_size = context['pareto_front'].shape[0]
    
    if front_size > 1: 
        # Estimate sparsity of the Pareto front using pairwise distances between points in objective space.
        pf_obj_vals = np.array([context['pareto_front'][i] for i in range(front_size)])
        
        # Compute average distance to nearest neighbor among pareto-front points
        dists_to_pf_nbrs = []
        if len(pf_obj_vals) > 1:
            from scipy.spatial.distance import cdist
            
            distances = cdist(pf_obj_vals, pf_obj_vals)
            
            np.fill_diagonal(distances, np.inf)

            min_dists = np.min(distances, axis=0)
            front_sparsity_score = float(np.mean(min_dists))
        else: 
            # If there's only one point in the pareto-front, assume some baseline sparseness
            front_sparsity_score = 1.0
            
    elif front_size == 1:
        front_sparsity_score = np.linalg.norm(context['pareto_front'][0] - ref_point)
        
    else: 
        # No observations yet; default to a small value for the sparse penalty.
        front_sparsity_score = 1e-3
    
    progress_factor = context["campaign"]["progress"]
    
    scores = []
    
    for i, cand in enumerate(context['pool']):
        acq_value_norm = float(cand.get('acq_value_norm',0.5))  
        
        # Compute normalized uncertainty based on the spread of objective predictions
        sigma_sum = sum([cand["gp_posterior"][name]["std"] / front_range[name] for name in names])
            
        novelty_score_i = novelty_scores[i]
      
        # Scale up or down exploration bonus depending on how sparse current Pareto-front is.
        # Sparse regions encourage more focused uncertainty-based search
        sparsity_adjustment_factor = 1.0 + np.log(1e-6+front_sparsity_score)
        
        scaled_uncertainty_bonus = sigma_sum * (sparsity_adjustment_factor) 
        
        if progress_factor > 0:
            exploitation_weight = max(.2, .8 - progress_factor*0.5 )
            
            # Blend between acquisition and uncertainty-adjusted novelty
            score_i = acq_value_norm*(exploitation_weight + scaled_uncertainty_bonus *(1-exploitation_weight))
        
        else: 
           score_i =  (acq_value_norm * max(.2, .8 - progress_factor*0.5) +
                       sigma_sum )
            
        scores.append(score_i)
    
    return scores