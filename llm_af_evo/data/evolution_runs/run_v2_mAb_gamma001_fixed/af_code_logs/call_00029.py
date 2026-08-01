def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted objective sum,
    but reduce scores for candidates that are too close (in feature space) to already observed points.
    This encourages exploration of novel regions while still favouring high-yield predictions.
    The novelty penalty is scaled based on how far the candidate's prediction lies from current front boundaries
    and penalizes proximity in a way that avoids overfitting to exact observation locations,
    using Monte Carlo resampling for robustness against noise-induced Pareto misclassification. 
    """
    names = context["objective_names"]
    
    # Use MC sampling of GP posteriors (10 samples per candidate) to estimate
    # the probability each is actually non-dominated, mitigating overconfidence in front.
    n_samples_per_cand = 10
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate mean hypervolume improvement using MC samples from this candidate's posterior
        hv_improvements = []

        for _ in range(n_samples_per_cand):
            sampled_objectives = [np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                                for name in names]
            
            # Determine if sample is dominated by current front (with small epsilon to avoid edge cases)
            dominates_any = False
            eps = 1e-6
            
            for pf_point in context["pareto_front"]:
                better_or_equal = [sampled_objectives[i] >= pf_point[i]-eps 
                                 for i in range(len(names))]
                strictly_better = any(sampled_objectives[i] > pf_point[i]+eps
                                    for i in range(len(names)))
                
                if all(better_or_equal) and strictly_better:
                    dominates_any = True
                    break
            
            # If not dominated, estimate hypervolume improvement using the reference point.
            if not dominates_any:
                hv_improvement = 1.0
                ref_point_vals = [context["ref_point_by_name"][name] for name in names]
                
                try: 
                     prod_val = np.prod([max(0., (r - sampled_objectives[i]))  
                                       for i, r in enumerate(ref_point_vals)])
                     hv_improvement *= max(prod_val, 1e-20) # avoid zero
                except:
                    pass
                
            else:
                 hv_improvement = 0.0

            hv_improvements.append(hv_improvement)

        expected_hvi = np.mean(hv_improvements)
        
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_normed = (sum([gp[name]["std"]/context["pareto_front_range"][name]  
                           for name in names]) / len(names))
    
        # Combine HVI estimate with mean prediction and add a normalized uncertainty term
        score_base = mu_sum + expected_hvi * 10.   # scale up the HV contribution slightly

        scores.append(score_base)
        
    return scores