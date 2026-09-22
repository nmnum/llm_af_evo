def score_pool(context):
    """Blend normalized acquisition value with an entropy-based diversity incentive derived from posterior covariance structure."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute entropy-like measure of uncertainty across objectives
    entropies = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        stds = [gp[name]["std"] for name in names]
        # Use geometric mean to capture joint uncertainty structure
        if all(s > 0 for s in stds):
            entropy = np.exp(np.sum(np.log(stds)) / len(names))
        else:
            entropy = 0.0
        entropies.append(entropy)
    
    max_entropy = max(entropies) + 1e-8
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        acq_norm = cand["acq_value_norm"]
        # Normalize by maximum entropy to prevent domination by low uncertainty candidates
        diversity_score = (entropies[i] / max_entropy)
        
        # Combine acquisition value with a multiplicative penalty based on proximity 
        # of the candidate's predicted objectives to existing Pareto front in objective space.
        gp = cand["gp_posterior"]
        pred_objs = np.array([gp[name]["mean"] for name in names])
        
        # Compute distance from reference point (inverted so higher is better)
        dist_to_ref = np.linalg.norm(pred_objs - ref_point) 
        if dist_to_ref > 0:
            front_similarity_score = 1. / (dist_to_ref + 1e-8)
        else:
            front_similarity_score = float('inf')
        
        # Final score: acquisition value scaled by inverse distance to reference
        scores.append(acq_norm * front_similarity_score)

    return scores