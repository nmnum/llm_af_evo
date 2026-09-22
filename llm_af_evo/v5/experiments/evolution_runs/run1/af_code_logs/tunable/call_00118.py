def score_pool(context):
    """Score candidates by acquisition value modulated with inverse novelty, where novel points are those far from already selected ones."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base scores: normalized hypervolume improvement estimates  
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Collect all previously observed points (including current front)
    X_observed = context["X_obs"]
        
    final_scores = []
    selected_indices = set()
    
    for i, _ in enumerate(context["pool"]):
        # Find the nearest already-selected point to this candidate
        cand_x = context["pool"][i]["x"]
            
        if len(selected_indices) == 0:
            min_dist_to_selected = float('inf')
        else: 
            selected_Xs = X_observed[list(selected_indices)]
            distances = np.linalg.norm(selected_Xs - cand_x, axis=1)
            min_dist_to_selected = np.min(distances)

        # Inverse novelty score (higher when farther from any picked point)  
        if min_dist_to_selected == 0:
            novel_score = float('inf') 
        else:   
            novel_score = 1. / min_dist_to_selected
            
        final_scores.append(base_scores[i] * novel_score)
        
    return list(final_scores)