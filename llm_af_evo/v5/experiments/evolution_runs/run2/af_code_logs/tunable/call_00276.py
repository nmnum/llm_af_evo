def score_pool(context):
    """Rank pool greedily by picking best candidates one at a time; reduce scores of remaining candidates based on proximity to already-picked ones using an exponential decay multiplier."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Base scoring with acquisition value and uncertainty
    base_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        acq_norm = cand["acq_value_norm"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 
        
        # Combine acquisition value and uncertainty (UCB-style but using normalized values from the pool directly as base score, not mean+uncertainty terms separately).
        combined_score = acq_norm + 0.5 * sigma_norm
        base_scores.append(combined_score)
    
    picked_indices = []
    final_scores = [float('-inf')] * len(context["pool"])
        
    for _ in range(len(context["pool"])):
        # Pick the highest scoring unpicked candidate 
        best_idx = -1  
        best_base = float('-inf')
        for i, (base_score, is_picked) in enumerate(zip(base_scores, [i in picked_indices for i in range(len(context["pool"]))])):
            if not is_picked and base_score > best_base:
                best_base = base_score
                best_idx = i
                
        # If no unpicked candidates left stop picking.
        if best_idx == -1: 
             break
            
        picked_indices.append(best_idx)
        
        # Compute the multiplier for all remaining (unpicked) candidate indices based on distance to this newly-picked one.  
        new_x = context["pool"][best_idx]["x"]
            
        multipliers = []
        for i in range(len(context["pool"])):
            if i == best_idx:
                continue
                
            other_cand = context["pool"][i]
                
            # Compute L2 distance between features (normalized to [0,1] scale)
            dist_x_squared = np.sum((new_x - other_cand["x"]) ** 2) 
            
            multiplier_val = max(0.0, 1.0 - np.exp(-dist_x_squared))
            multipliers.append(multiplier_val)

        # Update scores for unpicked candidates
        updated_scores_for_unpicked = []
        
        i_idx = 0  
        for j in range(len(context["pool"])):
            
             if j == best_idx:
                 final_scores[j] = base_scores[best_idx]
                 
             else: 
                  assert len(multipliers) > i_idx, "Mismatched lengths."
                  
                  # Apply the multiplier to reduce score of similar candidates.
                  reduced_score = multipliers[i_idx] * base_scores[j]

                  updated_scores_for_unpicked.append(reduced_score)
                      
                  final_scores[j] = reduced_score
                  i_idx += 1

    return list(final_scores)