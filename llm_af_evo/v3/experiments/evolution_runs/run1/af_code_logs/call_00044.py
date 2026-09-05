def score_pool(context):
    """Scores candidates by resampling observed data to estimate expected hypervolume improvement under noise, with progressive exploitation vs exploration weighting."""
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

        # Estimate hypervolume improvement potential by resampling from observed data
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point relative to ref_point and current front  
            if (cand_obj >= ref_point).all():
                hv_improvement = 0.0
            else:
                hypervolume_ref = max(0, cand_obj[0] - ref_point[0]) * max(0, cand_obj[1] - ref_point[1])
                
                # Check if dominated by current front 
                dominates_any_front = False  
                for pf in context["pareto_front"]:
                    if (cand_obj >= pf).all() and not (cand_obj == pf).any():
                        dominates_any_front = True
                        break
                        
                hv_improvement = hypervolume_ref * float(not dominates_any_front)
                
            hv_improvements.append(hv_improvement)

        score = np.mean(hv_improvements) 
        
        # Add a progressive exploitation vs exploration weight based on campaign progress 
        p = context["campaign"]["progress"]
        
        if len(X_obs) > 0:
            
            cand_x = cand["x"] 
            
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)

            # Encourage diversity by penalizing proximity to existing points
            novelty_penalty = max(0.0, (min_distance / 0.2) ** 3 * p )
            
            score -= novelty_penalty

        scores.append(score)

    return scores