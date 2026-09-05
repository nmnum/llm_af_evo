def score_pool(context):
    """Blend acquisition value with an entropy-based diversity bonus that rewards candidates varying significantly across objectives."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute per-candidate objective entropies to encourage diverse predictions
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Normalize the standard deviations by front range to make them comparable
        normalized_stds = stds / np.array([front_range[n] for n in names]) if all(front_range.values()) else stds
        
        # Compute entropy of predicted objectives (higher is more diverse)
        objective_entropy = -np.sum(normalized_stds * np.log(normalized_stds + 1e-8)) 

        score = cand["acq_value_norm"] + 0.25 * objective_entropy
        scores.append(score)

    return scores