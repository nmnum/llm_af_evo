def score_pool(context):
    """Score candidates based on expected hypervolume improvement averaged across bootstrapped Pareto fronts."""
    n_bootstrap = 10
    scores = np.zeros(len(context["pool"]))
    
    for _ in range(n_bootstrap):
        # Bootstrap sample from Y_obs
        indices = np.random.choice(len(context["Y_obs"]), size=len(context["Y_obs"]), replace=True)
        Y_sample = context["Y_obs"][indices]
        
        # Compute non-dominated points (Pareto front) of the sampled data
        pf_sample = Y_sample[np.array([i for i in range(len(Y_sample)) if all(not (Y_sample[j] > Y_sample[i]).all() or (Y_sample[j] >= Y_sample[i]).all() for j in range(len(Y_sample)) if j != i)])]
        
        # Score each candidate based on hypervolume improvement
        for i, cand in enumerate(context["pool"]):
            cand_obj = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
            scores[i] += hypervolume_improvement(cand_obj, pf_sample, context["ref_point"])
    
    return scores / n_bootstrap

def hypervolume_improvement(candidate_objectives, pareto_front, ref_point):
    """Compute hypervolume improvement of a candidate relative to a Pareto front."""
    if len(pareto_front) == 0:
        return hypercube_volume(candidate_objectives, ref_point)
    
    # Find the dominated region by the current Pareto front
    dominated = np.array([not (candidate_objectives >= pf).all() for pf in pareto_front])
    
    # If candidate is dominated, no improvement
    if all(dominated):
        return 0.0
    
    # Otherwise, compute hypervolume of the region dominated by candidate + Pareto front
    # This is a simplified version; full implementation would be more complex
    try:
        # Compute volume of hypercube from candidate to reference point
        vol = np.prod(ref_point - candidate_objectives)
        return max(0.0, vol)
    except:
        return 0.0