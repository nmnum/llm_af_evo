def score_pool(context):
    """Scores candidates by resampling observed data to estimate improvement under noise, with progressive exploitation-weighted uncertainty."""
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"] 
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples to estimate improvement
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvements  = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Resample the observed data to estimate how much better this candidate would be 
            sampled_Y_obs = Y_obs[np.random.choice(len(Y_obs), size=len(Y_obs))]
            temp_front = []
            for y in sampled_Y_obs:
                is_dominated = False
                dominated_by_new = None
                
                for i, pf in enumerate(temp_front):
                    if (y <= pf).all() and not (y == pf).any():
                        # y is dominated by an existing point 
                        is_dominated = True  
                        break
                        
                    elif (pf < y).all():  # PF dominates this new one
                        dominated_by_new = i
                
                if not is_dominated:
                    temp_front.append(y)
                    
            hv_improvement = 0.0
            
            for pf in context["pareto_front"]:
                hypervolume_pf = np.prod(ref_point - pf) 
                
            # This would require a full HV computation, but let's simplify with dominance count
            dominate_count = sum( 
                (cand_obj >= pf).all() and not (cand_obj == pf).any()
                for pf in context["pareto_front"]
            )
            
            hv_improvements.append(dominate_count / len(context["pareto_front"]))
        
        # Use the mean improvement potential
        score = np.mean(hv_improvements)
    
        scores.append(score)

    return scores