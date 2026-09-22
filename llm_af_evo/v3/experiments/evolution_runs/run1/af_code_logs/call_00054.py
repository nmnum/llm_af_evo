def score_pool(context):
    """Estimates dominance potential by resampling observed points under noise to infer candidate's Pareto optimality likelihood."""
    n_samples = 50
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the candidate’s GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate how often this sample would be Pareto-optimal 
        n_pareto_optimal = 0.0
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Check if candidate dominates any observed point (i.e., it's potentially better)
            is_dominant = True 
            for obs_point in context["Y_obs"]:
                if all(cand_obj <= obs_point) and not all(cand_obj == obs_point):
                    is_dominant = False
                    break
            
            # If candidate isn't dominated by any point, it could be Pareto-optimal  
            n_pareto_optimal += float(is_dominant)
        
        score = 1.0 * n_pareto_optimal / n_samples 
        scores.append(score)

    return scores