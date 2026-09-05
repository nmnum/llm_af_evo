def score_pool(context):
    """Blend acquisition value with uncertainty and diversity penalty inspired by both parent strategies."""
    names = context["objective_names"]
    scores = []
    
    # Collect all candidate x values for novelty calculation  
    X_obs = context["X_obs"] 
    pool_x = np.array([cand['x'] for cand in context["pool"]])
    
    if len(X_obs) == 0:
        # No observations yet, use acquisition + uncertainty
        base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
        
        for i, (base_score, cand) in enumerate(zip(base_scores, context["pool"])): 
            ucb_bonus = sum(cand["gp_posterior"][name]["std"] for name in names)
            scores.append(base_score + 0.1 * ucb_bonus)

    else:
        # Compute distances from each candidate to the nearest observed point
        min_dists_x = []
        
        if len(X_obs) > 0:  
            dists_to_all_points = np.linalg.norm(pool_x[:, None, :] - X_obs[None, :, :], axis=2)
            min_dist_per_candidate = np.min(dists_to_all_points, axis=1)
            
        else:
            # No observations yet
            min_dist_per_candidate = [float('inf')] * len(context["pool"])
        
        for i in range(len(context["pool"])):
            dist_x = float(min_dist_per_candidate[i])
                        
            base_score = context["pool"][i]["acq_value_norm"]
            
            ucb_bonus = sum(context["pool"][i]["gp_posterior"][name]["std"] 
                            for name in names)
                
            # Diversity penalty: reduce score if candidate is too close to observed points
            diversity_factor = np.exp(-dist_x)  # Closer -> smaller factor
            
            final_score = base_score + 0.1 * ucb_bonus - (1e-3 * dist_x / max(1e-6, context["pareto_front_range"]["f2"]))
            
            scores.append(final_score)
    
    return scores