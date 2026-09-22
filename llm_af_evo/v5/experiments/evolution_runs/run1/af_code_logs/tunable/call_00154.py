def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted quality; boost candidates near the Pareto front's boundary using gradient-based sensitivity."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    
    # Compute raw qualities (sum of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)

    q_min, q_max = min(qualities), max(qualities) 
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)

    # Normalize acquisition values
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Compute uncertainty-adjusted scores: quality * std_penalty 
    penalty_factor = 1.0
    
    score_list = []
    for i, (norm_q, acq) in enumerate(zip(norm_qualities, acq_values)):
        gp_posterior = context['pool'][i]['gp_posterior']
        
        # Compute uncertainty as sum of normalized standard deviations
        total_sigma_norm = sum(gp_posterior[name]["std"] / 
                              context["pareto_front_range"][name] for name in names)
            
        std_penalty = 1.0 - penalty_factor * (total_sigma_norm / len(names))
                
        score_list.append(acq + norm_q * max(0., min(std_penalty, 1.)))

    return score_list