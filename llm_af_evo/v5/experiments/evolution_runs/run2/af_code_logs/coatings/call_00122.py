def score_pool(context):
    """Estimate contribution to pareto front diversity by sampling noisy GP objectives and measuring how often candidates expand coverage of objective space."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]

    n_samples_per_candidate = 50
    
    diversity_gains = []
        
    # Sample from each candidate's GP posterior to estimate how often it would
    # improve the pareto coverage (i.e. not be dominated by current front)
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        samples = []

        for _ in range(n_samples_per_candidate): 
            sampled_objectives = []
                
            # Sample from the joint GP posterior
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]  
                    
                if sigma <= 0:   # Avoid invalid std values that could break sampling
                    sampled_val = mu
                else:
                    sampled_val = np.random.normal(mu, sigma)
                        
                sampled_objectives.append(sampled_val)

            samples.append(np.array(sampled_objectives))
        
        n_not_dominated_by_front = 0
        
        for sample in samples: 
                
            # Check if this noisy sample would be dominated by current front
            pf_points = context["pareto_front"]
            
            is_dominateable = False
            
            if len(pf_points) > 0:
                # A point dominates another iff it's >= in all objectives and strictly >
                # in at least one objective.
                
                for pt in pf_points: 
                    dominated_by_this_pt = True
                    strict_improvement =False
                    
                    for j, val in enumerate(sample):
                        if not (val <= pt[j] + 1e-8):   # allow numerical tolerance  
                            dominated_by_this_pt=False
                        
                        elif(val > pt[j]):
                            strict_improvement=True 
                            
                    if(dominated_by_this_pt and strict_improvement):
                         is_dominateable = True
                         break
            
            else:
                n_not_dominated_by_front += 1   # No front points, so always not dominated
                
            
            if(not (is_dominateable or len(pf_points)==0)):
                 n_not_dominated_by_front+=1
                    
        diversity_gain_prob = float(n_not_dominated_by_front) / n_samples_per_candidate
        diversity_gains.append(diversity_gain_prob)

    # Combine base acquisition score with estimated contribution to pareto front coverage 
    final_scores = []
        
    for i in range(len(acq_scores)):
            
        combined_score = (acq_scores[i] + 0.5 * np.clip(diversity_gains[i], 0., 1.) )
                
        final_scores.append(combined_score)
        
    return final_scores