def score_pool(context):
    """Blend acquisition value with a dynamic uncertainty term and front-density-aware novelty penalty to balance exploration and exploitation across the Pareto frontier."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]

    # Compute normalized distances from each candidate to pareto front
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand['gp_posterior'][name]["mean"] for name in names])
        
        # Calculate hypervolume contribution of this point relative to reference 
        if len(pf) > 0:
            hv_contribs = [max(1e-8, ref_point[i] - pf[:, i].min()) * max(1e-8, gp_mean[i]) for i in range(len(names))]
            front_dist_normed = np.prod(hv_contribs)
        else: 
            # If no pareto points yet seen (early stage), use uncertainty-based proxy
            sigma_sum = sum(cand['gp_posterior'][name]["std"] for name in names)  
            front_dist_normed = 1.0 - min(1., max(0., sigma_sum / len(names)))
            
        # Normalize acquisition value to [0,1] scale 
        acq_value_norm = cand["acq_value_norm"]
        
        # Use progress-aware uncertainty bonus (exploit early)
        prog = context['campaign']['progress']
        exploit_weight = 2. * max(0., min(1., 1 - prog))
                
        sigma_sum_normed = sum(cand['gp_posterior'][name]["std"] / 
                               context["pareto_front_range"][name] for name in names)
        
        uncertainty_bonus = exploit_weight * (sigma_sum_normed/len(names)) 
        
        # Combine acquisition, front contribution and uncertainty
        final_score = acq_value_norm + 0.3*front_dist_normed - 1e-6*(uncertainty_bonus**2) 
                
        scores.append(final_score)
        
    return scores