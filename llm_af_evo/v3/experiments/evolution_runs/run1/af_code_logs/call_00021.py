def score_pool(context):
    """Integrates UCB-style exploration with hypervolume improvement potential and novelty-based diversity."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples to estimate dominance
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        n_dominate_front  = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            dominate_count = sum( 
                (cand_obj >= pf).all() and not (cand_obj == pf).any()
                for pf in context["pareto_front"]
            )
            n_dominate_front += dominate_count
            
        hypervolume_score = 1.0 * n_dominate_front / n_samples
        ucb_explore = sum(gp[name]["mean"] + 0.5*gp[name]["std"]for name in names)
        
        # Novelty term: inverse of distance to nearest observed point  
        x_cand = cand["x"]
        if len(X_obs) == 0:
            novelty = 1e6
        else:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            novelty = 1. / (min_distance + 1e-9) if min_distance > 0 else 1e6
            
        scores.append(ucb_explore * hypervolume_score + 0.5*novelty)
    
    return scores