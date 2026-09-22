def score_pool(context):
    """Estimate probability of a candidate being Pareto optimal and blend with acquisition value for robust exploration-exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute hypervolume contributions of each candidate
    hv_contributions = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate probability that this candidate is Pareto optimal 
        # by computing how much it dominates the current front (in log space)
        dominated_count = 0  
        total_front_points = len(context['pareto_front'])
        
        if total_front_points > 0:
            for i, pf_point in enumerate(context["pareto_front"]):
                is_dominated = True
                # Check each objective: candidate must be >= front point on all objectives 
                # and strictly better than at least one to dominate it  
                dominates_any = False
                
                for j, name in enumerate(names):    
                    cand_val = gp_posterior[name]["mean"]
                    pf_val = pf_point[j]
                    
                    if cand_val < pf_val:
                        is_dominated = False
                        break
                        
                    elif cand_val > pf_val: 
                        dominates_any = True
                    
                # If candidate >= all objectives and strictly better on at least one, it dominates  
                if is_dominated and dominates_any:
                    dominated_count += 1
                
        prob_pareto_optimal = (total_front_points - dominated_count) / max(1.0, total_front_points)
        
        hv_contributions.append(prob_pareto_optimal * cand["acq_value_norm"])
    
    # Normalize contributions to [0, 1] range
    if len(hv_contributions) > 0:
        norm_factor = np.max(np.abs(hv_contributions)) 
        normalized_hvs = [h / max(1e-8, norm_factor) for h in hv_contributions]
        
        # Combine with acquisition value (re-weighted by progress)
        campaign_progress = context["campaign"]["progress"]
        acq_weight = 0.7 + 0.3 * campaign_progress
        unc_bonus = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
                
    else:
        normalized_hvs = [cand["acq_value_norm"] for cand in context["pool"]]
        
    scores = []
    
    # Blend acquisition value with HV contribution and uncertainty bonus
    ucb_weight = 0.5
    
    for i, (hv_contrib, cand) in enumerate(zip(normalized_hvs, context["pool"])): 
        gp_posterior = cand["gp_posterior"]
        unc_bonus = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        score = acq_weight * hv_contrib + \
                (1.0 - acq_weight) * cand['acq_value_norm'] + \
                 ucb_weight * unc_bonus
                 
        scores.append(score)

    return scores