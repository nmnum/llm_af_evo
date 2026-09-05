def score_pool(context):
    """Score candidates by expected hypervolume improvement adjusted for exploration-exploitation balance and posterior uncertainty scaling."""
    names = context["objective_names"]
    
    # Use the precomputed acquisition values directly 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize acquisition scores to [0, 1] range
    a_min, a_max = np.min(acq_values), np.max(acq_values)
    if abs(a_max - a_min) < 1e-9:
        norm_acqs = np.full_like(acq_values, 0.5)
    else: 
        norm_acqs = (acq_values - a_min)/(a_max-a_min)

    # Scale uncertainty based on campaign progress and stagnation
    step_frac = context["campaign"]["progress"]
    
    # Early exploration bias with decay  
    if step_frac < 0.3:
        ucb_weight = max(0., min((1.-step_frac)/0.2, 1.) * (1.+context['campaign']['stagnant_batches']*0.5))
    else: 
        ucb_weight = max(0., 1. - step_frac*0.8)
    
    # Incorporate uncertainty from GP posteriors
    uncertainties = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum_sq = sum(gp[name]["std"]**2 for name in names) 
        total_uncertainty = np.sqrt(sigma_sum_sq / len(names))  # average std dev per obj
        
        if step_frac > 0.7 and not context['campaign']['stagnant_batches']:
            ucb_weight *= (1.+total_uncertainty*5.)
        
        uncertainties.append(total_uncertainty)
    
    uncertainty_scores = -np.array(uncertainties) * ucb_weight

    # Final score: acquisition value weighted by scaled uncertainty
    scores = norm_acqs + 0.3 * uncertainty_scores
    
    return list(scores)