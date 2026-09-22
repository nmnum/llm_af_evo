def score_pool(context):
    """Score candidates by base acquisition value adjusted for diversity: pick best candidate greedily, then reduce scores of similar remaining candidates based on distance to already-picked ones."""
    names = context["objective_names"]
    
    # Start with the raw acq_value_norm as baseline score
    base_scores = [cand['acq_value_norm'] for cand in context['pool']]
    
    picked_indices = []
    final_scores = [-1.0] * len(context['pool'])
    
    while len(picked_indices) < len(context['pool']):
        # Pick the best unpicked candidate by base score
        max_idx = -1
        max_score = -float('inf')
        
        for i in range(len(base_scores)):
            if final_scores[i] == -1.0 and base_scores[i] > max_score:
                max_score = base_scores[i]
                max_idx = i
                
        picked_indices.append(max_idx)
                
        # Set the actual score of this candidate
        final_scores[max_idx] = base_scores[max_idx]

        # Now adjust scores for remaining candidates based on proximity to newly-picked one
        
        if len(picked_indices) == 1:
            continue
            
        prev_picked_x = context['pool'][picked_indices[-2]]["x"]
        
        for i in range(len(base_scores)):
            
            if final_scores[i] != -1.0: 
                # Already picked
                continue
                
            curr_cand_x = context['pool'][i]["x"] 
            
            dist_sq = np.sum((curr_cand_x - prev_picked_x) ** 2)
                
            multiplier = 1.0 - np.exp(-dist_sq)

            final_scores[i] *= multiplier
            
    return [s if s >= 0 else base for s,base in zip(final_scores, base_scores)]