def score_pool(context):
    """Score candidates by acquisition value enhanced with progressive uncertainty weighting and front-anchored exploitation."""
    names = context["objective_names"]
    
    # Base scores from normalized acq_value_norm 
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute GP-based quality metrics
    qualities = []
    uncertainties = []  
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names) 
        qualities.append(mu_sum)
        uncertainties.append(sigma_sum)

    # Normalize quality and uncertainty
    q_min, q_max = min(qualities), max(qualities)
    u_min, u_max = min(uncertainties), max(uncertainties)
    
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else: 
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)

    if abs(u_max - u_min) < 1e-9:
        norm_uncertainties = np.array([0.5] * len(uncertainties))  
    else:
        norm_uncertainties = (np.array(uncertainties) - u_min) / (u_max - u_min)

    # Progressive uncertainty weight: higher early, lower late
    progress = context["campaign"]["progress"]
    
    w_u = 0.5 * np.sin(np.pi/2 * min(progress*3., 1.) ) + 0.5
    
    # Exploitation signal from quality vs front range  
    ranges = [context['pareto_front_range'][name] for name in names]
    avg_range = sum(ranges) / len(ranges)
    
    exploitation_signal = np.array(norm_qualities)

    scores = []
    for i, (base_score, u_weighted) in enumerate(zip(base_scores, w_u * norm_uncertainties)):
        # Blend base acquisition with uncertainty and quality
        score_i = 0.7*base_score + 0.3*u_weighted 
        if progress > 0.5:
            # Add exploitation boost for candidates near front range  
            normalized_quality = (qualities[i] - q_min) / max(q_max-q_min,1e-9)
            
            score_i += 0.2 * np.clip(normalized_quality/avg_range , 0., 1.) 
        scores.append(score_i)

    return scores