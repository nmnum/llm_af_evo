def score_pool(context):
    """Score candidates by acquisition value adjusted for diversity against already-selected fronts and uncertainty normalization."""
    names = context["objective_names"]
    ref_point = context["ref_point"] 
    front = context['pareto_front']
    
    # Normalize acq values to [0, 1] range
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    if len(base_scores) > 1:
        scaled_acqs = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
    else:
        scaled_acqs = base_scores.copy()
        
    # Compute uncertainty-normalized scores
    unc_norms = []
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        sigma_sum = sum(gp_posterior[name]["std"] / (context["pareto_front_range"][name]) 
                        for name in names)
        unc_norms.append(sigma_sum)

    # Normalize uncertainty scores to [0, 1] range
    if len(unc_norms) > 1:
        scaled_uncert = np.array(unc_norms) - np.min(unc_norms)
        max_unc = np.max(scaled_uncert)
        unc_scores = (scaled_uncert / max_unc) if max_unc != 0 else np.zeros_like(scaled_uncert)
    else:
        unc_scores = [0.]
        
    # Combine acquisition and uncertainty scores
    combined_score = scaled_acqs + 1 * unc_scores
    
    return list(combined_score)