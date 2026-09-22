def modifier(context):
    """Add a correction term that rewards candidates with high predicted objective variance across objectives, promoting exploration of underexplored regions in outcome space."""
    names = context["objective_names"]
    values = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute the standard deviation of predictions across all objectives
        stds = [gp[name]["std"] for name in names]
        variance_score = np.var(stds)
        
        # Scale by how close we are to stagnation (higher correction when stagnant)
        stagnation_factor = min(1.0, context["campaign"]["stagnant_batches"] / 3.0) + 0.2
        
        values.append(variance_score * stagnation_factor)

    return values