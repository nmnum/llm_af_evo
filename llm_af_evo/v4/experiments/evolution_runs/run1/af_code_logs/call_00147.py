def score_pool(context):
    """Boost acquisition scores based on how much they extend the dominated hypervolume relative to current front spread."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute base acq values and uncertainty bonuses
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / (context["pareto_front_range"][name] or 1.0) 
            for name in names)
        for cand in context["pool"]
    ]
    
    # Normalize uncertainties to [0, 1]
    sigma_max = max(sigmas) if any(s > 0 for s in sigmars) else 1e-6
    
    scores = []
    for i, (acq, sigma) in enumerate(zip(acqs, sigmas)):
        normalized_sigma = sigma / sigma_max
        
        # Compute expected improvement relative to current front
        cand_means = np.array([cand["gp_posterior"][name]["mean"] 
                               for name in names])
        
        if len(context["pareto_front"]) == 0:
            hypervolume_gain_factor = acq * (1. + normalized_sigma)
            
        else:    
            # Estimate how much this candidate would improve the front's coverage
            dominated_by_current_pf = [
                all(cand_means[j] <= context["pareto_front"][i][j]
                    for j in range(len(names)))
                for i in range(len(context["pareto_front"]))
            ]
            
            if any(dominated_by_current_pf):
                # If candidate is dominated, its gain contribution to HV would be small
                hypervolume_gain_factor = 0.5 * acq
                
            else:
                # Otherwise it may improve the front's extent in objective space 
                ref_dist_to_cand_means = np.linalg.norm(ref_point - cand_means)
                
                if not any(np.isinf(ref_dist_to_cand_means)):
                    hv_contribution_ratio = min(1., 2. * (ref_dist_to_cand_means / max(
                        [np.linalg.norm(ref_point - pf) for pf in context["pareto_front"]]
                     ) or 1e-6)))
                else:
                    hv_contribution_ratio = acq
                
                hypervolume_gain_factor = acq + normalized_sigma *hv_contribution_ratio

        scores.append(hypervolume_gain_factor)
    
    return scores