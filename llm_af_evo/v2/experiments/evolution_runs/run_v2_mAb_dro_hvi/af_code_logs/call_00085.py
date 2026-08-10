def score_pool(context):
    """
    Exploitation with probabilistic pareto dominance: rank by predicted objective sum,
    adjusted by how likely each candidate is to be on the Pareto front based on GP uncertainty.
    This uses a sampling-based estimate of posterior probability that a point dominates none others, 
    encouraging exploration while still exploiting high-performing candidates.  
    """
    names = context["objective_names"]
    scores = []
    
    # Sample from each candidate's posterior to estimate dominance likelihood
    n_samples = 100
    dominating_probs = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples for this candidate across all objectives  
        samples = np.array([
            np.random.normal(gp[name]["mean"], gp[name]["std"], size=n_samples)
            for name in names]).T

        # Count how many times each sample dominates the current Pareto front
        n_dominating = 0.
        
        if len(context["pareto_front"]) > 0:
            pf = context["pareto_front"]
            
            # For this candidate's samples, count which ones dominate any point in PF  
            for s in samples: 
                dominates_any_pf_point = False
                for pf_row in pf:
                    if all(s[i] >= pf_row[i] for i in range(len(names))):
                        dominates_any_pf_point = True
                        break
                        
                n_dominating += float(dominates_any_pf_point)
        else:
            # No front yet — assume this is a good candidate  
            n_dominating = 1.0
            
        prob_dominate_front = max(1e-6, min(n_dominating / len(samples), 1 - 1e-6))
        
        dominating_probs.append(prob_dominate_front)

    # Combine mean prediction with dominance probability 
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        p_dominant = dominating_probs[i]
        
        scores.append(mu_sum * np.log(1 + 5*p_dominant)) 

    return scores