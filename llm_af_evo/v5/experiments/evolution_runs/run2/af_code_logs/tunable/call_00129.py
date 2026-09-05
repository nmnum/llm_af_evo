def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and novelty reward."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distance to nearest previously observed point (novelty)
    X_obs = context["X_obs"]
    novelty_scores = []
    
    if len(X_obs) > 0:
        for i, cand in enumerate(context["pool"]):
            x_cand = cand['x']
            
            distances = np.linalg.norm(X_obs - x_cand, axis=1)

            min_distance = np.min(distances)
                        
            # Invert to get novelty score (higher is more novel)  
            if min_distance == 0:
                nov_score = float('-inf')   # Avoid exact matches
            else: 
                nov_score = 1. / (min_distance + 1e-8)

            novelty_scores.append(nov_score)
    else:
        novelty_scores = [0.] * len(context["pool"])

    # Progress-aware uncertainty weighting based on campaign state  
    progress = context['campaign']['progress']
    
    # Early: favour exploration; late: exploit more
    w_acq = 1. - np.clip(progress, 0., 1.) ** 2
    
    # Uncertainty weight decreases as we get closer to the end 
    u_weight = (1.- progress) * 0.5 + 0.3 
    
    scores = []
    
    for i in range(len(context["pool"])):
        cand_acq = acq_scores[i]
        
        # Add uncertainty term that scales with campaign stage  
        gp_posterior = context['pool'][i]["gp_posterior"]
                
        total_uncertainty = sum(gp_posterior[name]['std'] 
                                for name in names)
                                
        u_term  = (u_weight * total_uncertainty) 

        cand_score = w_acq * cand_acq + \
                     (1. - w_acq)* np.clip(u_term,0.,2.) +\
                      0.3* novelty_scores[i]
        
        scores.append(cand_score)
    
    return scores