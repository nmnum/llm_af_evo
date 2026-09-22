def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    acq_norm = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):  # No signal available
        return acq_norm
    
    names = context["objective_names"]
    corr_key = ",".join(names)
    correlations = context["obj_correlation"].get(corr_key, [0.] * len(context["pool"]))
    
    for i in range(len(context["pool"])):
        bonus = max(0., -correlations[i])  # Positive contribution only from negative correlation
        scores.append(acq_norm[i] + 0.1 * bonus)
        
    return scores