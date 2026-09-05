def score_pool(context):
    """Rank candidates greedily by base acquisition value, then reduce scores of nearby contenders to encourage diversity."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    pool_size = len(context["pool"])
    picked_indices = []  # indices of candidates already selected
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    final_scores = [-1.0] * pool_size
    
    def x_distance(i, j):
        return np.linalg.norm(context["pool"][i]["x"] - context["pool"][j]["x"])
    
    def objective_distance(i, j):
        dists = [abs(context["pool"][i]["gp_posterior"][name]['mean'] 
                     - context["pool"][j]["gp_posterior"][name]['mean']) / front_range[name]  
                 for name in names]
        return np.mean(dists)
    
    # Greedily pick candidates
    remaining_indices = list(range(pool_size))
    while len(picked_indices) < pool_size:
        best_idx = -1
        best_base_score = float('-inf')
        
        # Find the highest base score among unselected items
        for idx in remaining_indices:
            if base_scores[idx] > best_base_score and not (idx in picked_indices):
                best_base_score = base_scores[idx]
                best_idx = idx
        
        if best_idx == -1:  # Shouldn't happen, but just to be safe.
            break
            
        picked_indices.append(best_idx)
        
        multiplier = 1.0
        for old_pick_i in range(len(picked_indices)-1):
            pick_idx = picked_indices[old_pick_i]
            
            x_dist = x_distance(best_idx, pick_idx) 
            obj_dist = objective_distance(best_idx, pick_idx)

            # Combine distances (you can adjust how much each contributes)
            combined_dist = 0.5 * x_dist + 0.5 * obj_dist
            
            multiplier *= np.exp(-combined_dist)
            
        final_scores[best_idx] = base_scores[best_idx] * max(1e-6, multiplier) 
        remaining_indices.remove(best_idx)

    return [s if s >= -0.9 else float('-inf') for s in final_scores ]  # Ensure no negative scores except possibly very small ones