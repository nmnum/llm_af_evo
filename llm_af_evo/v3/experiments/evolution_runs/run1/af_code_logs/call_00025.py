def score_pool(context):
    """Adaptive UCB with hypervolume potential and novelty bonus, balancing exploitation, uncertainty, and progress-aware exploration."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamically adjust the weight between mean and std based on progression
    w_exploit = 0.3 + 0.7 * progress

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 

        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # UCB-style score: weighted mean and normalized std
        ucb_score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm

        # Estimate hypervolume improvement potential via sampling 
        n_samples = 30  
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Count how often this candidate dominates the current front
        n_dominate_front = 0.0 
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            dominate_count = sum(  
                (cand_obj >= pf).all() and not (cand_obj == pf).any()
                for pf in context["pareto_front"]
            )
            n_dominate_front += dominate_count

        hv_potential = 0.5 * n_dominate_front / n_samples 

        # Novelty term based on minimum distance to observed points
        novelty = np.linalg.norm(X_obs - cand["x"], axis=1).min()

        scores.append(ucb_score + 2.3648 * hv_potential + 0.9751 * novelty)

    return scores