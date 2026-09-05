def score_pool(context):
    """Blend acquisition value with an entropy-based exploration bonus and a front-density-adjusted exploitation signal."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    pf = context["pareto_front"]
    
    # Compute base scores using acq_value_norm
    raw_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    if len(pf) == 0:
        return raw_scores

    front_range = np.array([context["pareto_front_range"][name] for name in names])
    # Normalize pareto points to [0,1]
    pf_normalized = (pf - ref_point) / front_range
    
    scores = []
    
    for cand in context["pool"]:
        gp = cand['gp_posterior']
        
        mu_vec = np.array([gp[name]["mean"] for name in names])
        sigma_diag = np.array([gp[name]["std"] for name in names])

        # Compute entropy bonus: sum of log(variance) across objectives
        eps = 1e-8  
        ent_bonus = -np.sum(np.log(sigma_diag ** 2 + eps))
        
        # Normalize mu_vec to [0,1] using ref_point and front_range 
        cand_normalized = (mu_vec - ref_point)/front_range
        
        # Compute squared Mahalanobis distance from candidate mean vector
        dist_sq_mahal = np.sum(((cand_normalized - pf_normalized) / front_range)**2 , axis=1)
        
        if len(dist_sq_mahal) > 0:
            min_dist_to_front_squared = np.min(dist_sq_mahal)
            
            # Adjust exploitation score based on density near candidate
            exp_factor = (min_dist_to_front_squared + eps ) ** (-0.5)

            final_score = raw_scores[context["pool"].index(cand)] * 1e-3*exp_factor 
        else:
             final_score =raw_scores[context["pool"].index(cand)]

        scores.append(final_score)
    
    return [s + ent_bonus for s in scores]