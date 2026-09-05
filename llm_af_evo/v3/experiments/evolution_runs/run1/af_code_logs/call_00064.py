def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to assess Pareto dominance probability."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate the candidate's chance of being on or improving the front
        n_samples = 100
        dominates_count = 0
        
        # Sample from this candidate’s GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        for f1_samp, f2_samp in zip(samples_f1, samples_f2):
            # Check if this sample dominates any point on the current Pareto front
            is_dominant = True  # assume dominant until proven otherwise
            
            for pf_point in pareto_front:
                if (f1_samp <= pf_point[0] and f2_samp <= pf_point[1]):
                    # This candidate's sample does not dominate this PF point 
                    continue  
                
                elif (f1_samp >= pf_point[0]) or (f2_samp >= pf_point[1]):   
                     is_dominant = False
                     break

            if is_dominant:
                 dominates_count += 1
                
        # Score based on probability of dominance, with bonus for uncertainty  
        prob_dominate = dominates_count / n_samples 
        score = (prob_dominate + gp["f1"]["std"] *gp ["f2"]["std"]) 
        
        scores.append(score)
        
    return scores