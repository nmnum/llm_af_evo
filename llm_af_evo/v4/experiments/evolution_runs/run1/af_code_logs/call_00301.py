def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_key_fmt = "{},{}"
    
    # Build the correlation signal
    bonus_terms = np.zeros(len(context["pool"]))
    if len(names) >= 2:
        for i, _ in enumerate(context["pool"]):
            total_corr = 0.0
            count = 0
            
            for a_idx, name_a in enumerate(names):
                for b_idx, name_b in enumerate(names):
                    if a_idx != b_idx:
                        key = corr_key_fmt.format(name_a, name_b)
                        if key in context["obj_correlation"]:
                            total_corr += max(0.0, -context["obj_correlation"][key][i])
                            count += 1
            
            bonus_terms[i] = (total_corr / count) if count > 0 else 0.0
    
    # Blend the acquisition value with correlation-based bonus
    for i in range(len(context["pool"])):
        base_score = context["pool"][i]["acq_value_norm"]
        adjusted_bonus = 0.1 * bonus_terms[i] 
        scores.append(base_score + adjusted_bonus)
    
    return scores