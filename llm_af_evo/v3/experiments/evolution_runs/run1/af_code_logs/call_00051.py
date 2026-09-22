def score_pool(context):
    """Estimates candidate dominance likelihood using GP samples to guide exploration towards unexplored non-dominated regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Sample from each candidate's posterior to estimate Pareto membership probability  
    n_samples = 50
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], n_samples) 
        
        # Count how many sampled points are non-dominated by current front
        dominates_count = 0 
        for s_f1, s_f2 in zip(samples_f1, samples_f2):
            is_nondominated = True  
            for pf_point in context["pareto_front"]:
                if (s_f1 <= pf_point[0] and s_f2 <= pf_point[1]):
                    # dominated by this point
                    is_nondominated = False 
                    break
            
            if is_nondominated:
                dominates_count += 1
        
        # Score: probability of being non-dominated + normalized uncertainty  
        prob_nondom = float(dominates_count) / n_samples
        sigma_norm = (gp["f1"]["std"] + gp["f2"]["std"]) / sum(front_range.values())
        
        scores.append(prob_nondom * 0.7 + sigma_norm * 0.3)

    return scores