def score_pool(context):
    """Blend acquisition value with uncertainty, but suppress scores for near-duplicates based on observed points."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    x_obs = context["X_obs"] 
    acq_weights = [0.75, 0.25] # weight acquisition vs novelty
    ucb_weight = 0.1
    
    scores = []  
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Base score from botorch's qLogNEHVI 
        acq_value = cand["acq_value_norm"]

        # UCB-style uncertainty bonus
        ucb_bonus = sum(gp_posterior[name]["std"] for name in names) * ucb_weight

        # Novelty: inverse distance to nearest observed point  
        x_candidate = cand["x"]
        
        if len(x_obs) > 0:
            distances = np.linalg.norm(x_obs - x_candidate, axis=1)
            min_distance = np.min(distances)
            
            # Inverse of squared distance (to avoid division by zero), scaled
            novelty_score = 1. / (min_distance * min_distance + 1e-6) if min_distance > 0 else float('inf')
        else:
            novelty_score = 1.

        combined_score = acq_weights[0] * acq_value + \
                         acq_weights[1] * novelty_score + \
                         ucb_bonus

        scores.append(combined_score)
    
    # Suppress score of candidates that are too close to already selected ones (proximity-based suppression) 
    threshold = 0.2
    suppressed_scores = []
    for i, _ in enumerate(context["pool"]):
        
        current_x = context["pool"][i]["x"]
        is_suppressed = False
        
        # Check against previously picked candidates from this batch if any were already selected  
        try:
            prev_selected_idx = [j for j,score in enumerate(suppressed_scores) 
                                if score > 0 and i != j]
            
            distances_to_prevs = np.linalg.norm(
                current_x - context["pool"][prev_selected_idx]["x"], axis=1
            )
        
        except:
            # No previous selection or other error, proceed normally  
            suppressed_scores.append(scores[i])
            continue
            
        if len(distances_to_prevs) > 0 and any(d < threshold for d in distances_to_prevs):
             is_suppressed = True
                
        final_score = scores[i] * (1. - float(is_suppressed)) 
                
        # Add a small penalty to the score of candidates that are too close
        suppressed_scores.append(final_score)
        
    return [float(score) for score in suppressed_scores if not np.isnan(float(score))]