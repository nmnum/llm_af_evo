def score_pool(context):
    """Estimate probabilistic pareto optimality by perturbing candidates' GP posteriors and measuring how often they dominate."""
    names = context["objective_names"]
    scores = []
    
    # Use a small noise level for perturbation (standard deviation of the GP)
    eps = 1e-4
    
    n_samples = 20
    dominance_counts = np.zeros(len(context["pool"]))
    
    for _ in range(n_samples):
        # Sample from each candidate's joint posterior distribution with some jitter
        for i, cand in enumerate(context["pool"]): 
            gp_posterior = cand["gp_posterior"]
            
            # Draw noisy samples: mean + noise * std (noise ~ N(0,1))
            sample_f1 = np.random.normal(gp_posterior['f1']['mean'], eps if not gp_posterior['f1']['std'] else gp_posterior['f1']['std'])
            sample_f2 = np.random.normal(gp_posterior['f2']['mean'], eps if not gp_posterior['f2']['std'] else gp_posterior['f2']['std'])

            # Check dominance against current Pareto front
            is_dominated_by_front = False 
            for pf_point in context["pareto_front"]:
                dominates_pf = (sample_f1 >= pf_point[0]) and (sample_f2 > pf_point[1])
                
                if not dominates_pf:
                    continue
                    
                # If this sample point strictly dominates a front member, then it's potentially good
                is_dominated_by_front = True
                
            dominance_counts[i] += int(is_dominated_by_front)
    
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Combine acquisition score with the estimated probabilistic Pareto-optimality bonus  
    scores = [
        acq + 0.1 * (dominance_counts[i] / n_samples) 
        for i, acq in enumerate(base_scores)
    ]
        
    return scores