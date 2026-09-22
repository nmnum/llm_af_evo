def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    acq_norm = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):  
        return acq_norm
    
    names = context["objective_names"]
    corr_bonus = np.zeros(len(context["pool"]))
    
    # Collect correlation signals between pairs involving objectives
    corrs_by_candidate = {}
    for key, values in context["obj_correlation"].items():
        if "," not in key:
            continue  
        a_name, b_name = key.split(",", 1)
        if a_name in names and b_name in names: 
            corrs_by_candidate[(a_name, b_name)] = np.array(values)

    # For each candidate compute average negative correlation with objectives near the front
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Identify which objectives are currently well-covered by pareto front (near bounds)
        covered_obj_names = []
        for name in names:
            obj_range = context['pareto_front_range'][name]
            if not np.isclose(obj_range, 0.):
                # Consider objective 'covered' when its current value is near the boundary
                min_val = cand["gp_posterior"][name]["mean"] - 3 * cand["gp_posterior"][name]["std"]
                max_val = cand["gp_posterior"][name]["mean"] + 3 * cand["gp_posterior"][name]["std"]

                front_min, front_max = np.min(context['pareto_front'][:, names.index(name)]), \
                                       np.max(context['pareto_front'][:, names.index(name)])

                if min_val < (front_min + obj_range / 5) or max_val > (front_max - obj_range / 5):
                    covered_obj_names.append(name)

        # If no objectives are considered "covered", skip bonus
        if not covered_obj_names:
            continue

        total_corr = []
        
        for a_name in names:  
            
            neg_correlations_for_a = [] 
            
            for b_name in covered_obj_names:

                corr_key1, corr_key2 = (a_name,b_name), (b_name,a_name)
                
                if corr_key1 in corrs_by_candidate:
                    val = float(corrs_by_candidate[corr_key1][i])
                    
                elif corr_key2 in corrs_by_candidate: 
                    # Flip sign for correlation matrix symmetry
                    val = -float(corrs_by_candidate[corr_key2][i]) 

                else:
                    continue

                if val < 0. and a_name != b_name:
                    neg_correlations_for_a.append(val)

            avg_neg_corr = np.mean(neg_correlations_for_a) if len(neg_correlations_for_a)>1 \
                           else (neg_correlations_for_a[0] if neg_correlations_for_a else 0.)

            total_corr.append(avg_neg_corr)
            
        corr_bonus[i] = max(0., -np.sum(total_corr))

    # Combine acquisition value with bonus term
    final_scores = np.array(acq_norm) + 0.1 * (corr_bonus / (max(np.max(corr_bonus), 1e-8)))
    
    return list(final_scores)