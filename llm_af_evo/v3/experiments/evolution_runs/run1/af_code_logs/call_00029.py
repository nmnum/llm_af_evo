def score_pool(context):
    """Estimates improvement potential by resampling candidates' posteriors and measuring how often they dominate or are dominated, with early exploration bias."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    progress = context["campaign"]["progress"]

    # Early emphasis on uncertainty
    w_uncert = 0.5 * (1 - np.exp(-3 * progress))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the candidate's own posterior to estimate dominance probability 
        n_samples = 20
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate probability of being Pareto-optimal (not dominated by current front)  
        dominates_count = 0
        for s in range(n_samples):
            is_dominated_by_front = False 
            f1_s, f2_s = samples_f1[s], samples_f2[s]
            
            if len(context["pareto_front"]) > 0:
                # Check against all non-dominated points  
                for pf_point in context["pareto_front"]:
                    if (f1_s <= pf_point[0] and f2_s < pf_point[1]) or \
                       (f1_s < pf_point[0] and f2_s <= pf_point[1]):
                        is_dominated_by_front = True
                        break
            
            # If not dominated by front, count as potential improvement  
            if not is_dominated_by_front:
                dominates_count += 1

        dominance_prob = float(dominates_count) / n_samples
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        score = w_uncert * (mu_sum + gp["f2"]["std"]) + (1 - w_uncert) * dominance_prob
        scores.append(score)

    return scores