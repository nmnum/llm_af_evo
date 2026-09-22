def score_pool(context):
    """Score candidates by normalized acquisition value enhanced with an entropy-based diversity bonus that penalizes overexposure to any single objective's uncertainty pattern."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute the average standard deviation across objectives for each candidate
    sigma_avg = np.array([cand["gp_posterior"][name]["std"] 
                          for cand in context["pool"] 
                          for name in names]).reshape(len(context["pool"]), -1).mean(axis=1)
    
    # Normalize entropy of uncertainty pattern (higher entropy implies more balanced exploration across objectives)  
    eps = 1e-8
    sigma_normed = np.array([cand["gp_posterior"][name]["std"] / front_range[name] 
                             for cand in context["pool"] 
                             for name in names]).reshape(len(context["pool"]), -1)
    
    # Compute entropy: H(X) = -sum(p_i * log p_i), where p_i is normalized std
    sigma_probs = np.maximum(sigma_normed, eps)
    sigma_probs /= sigma_probs.sum(axis=1, keepdims=True)
    entropies = -(sigma_probs * np.log(sigma_probs)).sum(axis=1)

    # Final score: acquisition value plus entropy bonus (normalized to [0, 1] range of acq values) 
    scores = []
    for i, cand in enumerate(context["pool"]):
        base_score = cand['acq_value_norm']
        ent_bonus = np.clip(entropies[i], 0.0, 1.0)
        
        # Blend with a small weight to avoid overfitting
        final_score = base_score + 0.2 * ent_bonus 
        scores.append(final_score)

    return scores