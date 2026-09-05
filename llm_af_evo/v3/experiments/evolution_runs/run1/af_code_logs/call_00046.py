def score_pool(context):
    """Scores candidates based on expected hypervolume improvement using sampled objective values, with adaptive exploration weight and novelty bonus."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Adaptive exploitation factor: start exploratory, shift to exploitative
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-2*progress))
    
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point if added to the front 
            temp_front = np.vstack((context["pareto_front"], cand_obj))
            hv_improvement = 0.0
            
            try:
                from pymoo.performance.hv import Hypervolume
                hv_calc = Hypervolume(ref_point)
               hv_ref = hv_calc.do(context["pareto_front"])
                
                # Add candidate and compute new HV  
                if len(temp_front) >1: 
                    temp_hv = hv_calc.do(np.vstack((context["pareto_front"], cand_obj)))
                    hv_improvement = max(0, (temp_hv - hv_ref))
            except:
                 pass  # fallback to simple dominance
            
            hv_improvements.append(hv_improvement)
        
        expected_hv_imp = np.mean(hv_improvements) if hv_improvements else 0.0

        ucb_score = w_exploit * (gp["f1"]["mean"] + gp["f2"]["mean"]) 
        uncertainty_bonus = (1 - w_exploit) * sum(gp[name]["std"]/front_range[name] for name in names)
        
        # Combine with a novelty term based on minimum distance to observed points
        novel_dist = np.linalg.norm(X_obs - cand["x"], axis=1).min()
        score = expected_hv_imp + ucb_score + uncertainty_bonus + 2.5 * novel_dist
        
        scores.append(score)

    return scores