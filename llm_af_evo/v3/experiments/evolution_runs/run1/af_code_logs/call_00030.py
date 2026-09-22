def score_pool(context):
    """Scores candidates based on how often they would dominate the current front under sampled noise, with adaptive exploitation-uncertainty balance."""
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
        n_dominated_by_front, n_dominate_front  = 0.0, 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Check how often this sample dominates the current front
            dominate_count = sum( 
                (cand_obj >= pf).all() and not (cand_obj == pf).any()
                for pf in context["pareto_front"]
            )
            n_dominate_front += dominate_count
            
        score = 1.0 * n_dominate_front / n_samples
        scores.append(score)
    
    return scores