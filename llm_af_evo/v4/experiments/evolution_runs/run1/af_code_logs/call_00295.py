def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted progress awareness to favor candidates that are both promising and sufficiently novel."""
    
    if not context["pool"]:
        return []
        
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point_by_name"]

    # Compute base acquisition score
    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Normalize uncertainty relative to pareto front range  
    uncertainties = []
    for cand in context["pool"]:
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        uncertainties.append(sigma_sum)

    # Progress-aware component: how much better is this candidate's prediction than current ref point?
    progress_scores = []
    for cand in context["pool"]:
        pred_means = [cand['gp_posterior'][name]['mean'] for name in names]
        # Use reference point to compute normalized distance from dominated region
        dist_to_ref = np.array([max(0, ref_point[name] - mean) 
                               for (name, mean) in zip(names, pred_means)])
        progress_score = 1.0 / (1e-8 + np.mean(dist_to_ref))
        # Normalize to [0,1]
        normalized_progress = min(progress_score/np.max([p[2]*5 for p in [(ref_point[name], front_range[name]) 
                                                                          for name in names]]), 1)
        
        progress_scores.append(normalized_progress)

    scores = []
    for i, (acq_val, uncertainty, prog) in enumerate(zip(acq_values, uncertainties, progress_scores)):
        # Combine acquisition value with adjusted progress and inverse uncertainty
        score = acq_val + max(0.5 * prog - 1e-6*uncertainty**2 , 0)
        
        scores.append(score)

    return scores