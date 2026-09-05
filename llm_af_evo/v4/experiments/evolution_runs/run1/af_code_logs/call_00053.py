def score_pool(context):
    """Use acquisition value as primary signal and inject uncertainty-based exploration only when front is stagnant."""
    scores = []
    names = context["objective_names"]
    
    # Early termination: if not stagnating, just use raw acq values (exploit)
    campaign = context["campaign"] 
    if campaign["stagnant_batches"] == 0:
        return [cand["acq_value_norm"] for cand in context["pool"]]
        
    # When stagnant, blend acquisition with uncertainty-aware novelty
    front_range = context["pareto_front_range"]
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]

        # Compute normalized uncertainty (novelty bonus)
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names) 
        
        # Use a decaying factor based on stagnation batches
        decay_factor = 1.0 - min(0.9, campaign["stagnant_batches"] * 0.1)
        
        score = acq + sigma_sum * decay_factor
        
        scores.append(score)

    return scores