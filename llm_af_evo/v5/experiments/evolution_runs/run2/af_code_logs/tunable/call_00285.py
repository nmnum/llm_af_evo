def score_pool(context):
    """Blend acquisition value with a dynamic progress-aware uncertainty term that encourages exploration of under-explored objective regions based on candidate posterior variance and campaign stagnation level."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute base score from acq_value_norm
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Normalize the acquisition values to be between 0 and 1 (already done)
    max_acq, min_acq = np.max(base_scores), np.min(base_scores)
    if max_acq > min_acq:
        norm_base = [(s - min_acq) / (max_acq - min_acq) for s in base_scores]
    else:
        # All acquisition values are the same
        norm_base = [0.0] * len(context["pool"])
    
    campaign_progress = context['campaign']['progress']
    stagnation_level = context['campaign']['stagnant_batches']

    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Compute uncertainty as sum of normalized standard deviations
        total_uncertainty = np.sum(gp[name]["std"]/front_range[name] for name in names)
                
        # Dynamic weight based on progress and stagnation: more exploration early or when stagnant  
        w_explore = 0.5 * (1 - campaign_progress) + min(2, stagnation_level / 3.)
        
        score = norm_base[i]
        if not np.isclose(w_explore, 0):
            # Blend with uncertainty term
            normalized_uncertainty = total_uncertainty 
            adjusted_score = w_explore * normalized_uncertainty  
            final_score = (1 - w_explore) * score +adjusted_score
        
        else:
             final_score =score
            
        
        scores.append(final_score)
    
    return scores