def score_pool(context):
    """Blend acquisition value with dynamic uncertainty and Pareto probability for robust exploration-exploitation balance."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Dynamic UCB bonus that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    ucb_bonus_weight = 0.5 * (1 - progress)
    front_range = context["pareto_front_range"]
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus_weight * sigma_norm_sum)

    # Estimate probability of being Pareto-optimal using GP posteriors
    pareto_probs = []
    total_evals = 100
    
    if len(context["pareto_front"]) > 0:
        for i, cand in enumerate(context["pool"]):
            gp_posterior = cand["gp_posterior"]
            
            n_better_than_pf = 0
            
            # Sample from the GP posteriors to estimate Pareto probability
            for _ in range(total_evals):  
                sampled_vals = []
                
                for name in names:
                    mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                    
                    sampled_val = np.random.normal(mu, sigma)
                    sampled_vals.append(sampled_val)

                # Check if this sample is better than any point on the current Pareto front
                dominates_any_pf_point = False
                
                for pf_point in context["pareto_front"]:
                   (pf_dominates_sample) = all(
                        s >= pfp for (s, pfp) in zip(sampled_vals, pf_point)
                    )
                    
                    if pf_dominates_sample:
                        dominates_any_pf_point = True
                        break
                        
            pareto_probs.append(1.0 - n_better_than_pf / total_evals)

    else:  # No front yet  
        pareto_probs = [1.0] * len(context["pool"])
        
    final_scores = acq_scores + np.array(unc_scores) + \
                   0.3 * np.clip(np.array(pareto_probs), 0., 1.)

    return list(final_scores)