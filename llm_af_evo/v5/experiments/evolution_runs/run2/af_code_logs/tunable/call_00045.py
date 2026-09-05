def score_pool(context):
    """Estimate probability of a candidate being Pareto optimal and blend with acquisition value for robust exploration-exploitation."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate probability of being Pareto-optimal using GP posteriors
    pareto_probs = []
    front_range = context["pareto_front_range"]
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        # For each objective, compute how many standard deviations the mean is 
        # from the current Pareto frontier (assuming maximization)
        n_better_than_pf = 0
        total_evals = 1000

        for _ in range(total_evals):
            sample_means = []
            
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                
                # Sample from the GP posterior (assuming normality)
                sampled_val = np.random.normal(mu, sigma) 
                
                if len(context["pareto_front"]) > 0:  
                    pf_vals = context["pareto_front"][:, names.index(name)]
                    
                    # How many points in PF is this sample better than?
                    n_better_than_point_in_pf = sum(1 for val in pf_vals if sampled_val >= val)
                else:
                    # No front yet, assume it's likely to be good
                    n_better_than_point_in_pf = 0
                    
            pareto_probs.append(n_better_than_point_in_pf / len(context["pareto_front"]) 
                                if len(context["pareto_front"]) > 0 else 1.0)
    
    # Combine base acquisition score with estimated Pareto probability  
    final_scores = []
        
    for i in range(len(acq_scores)):
        prob_pareto = pareto_probs[i]
            
        combined_score = (acq_scores[i] 
                          + 0.2724 * np.clip(prob_pareto, 0., 1.) # weight Pareto estimate
                         )
                
        final_scores.append(combined_score)
        
    return final_scores