def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance checking."""
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Sample from GP posterior per objective
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in context["objective_names"]]
            for _ in range(n_samples)
        ])

        if len(front) == 0:
            # No front yet, all samples contribute full volume
            improvement_values = np.prod(np.maximum(samples - ref_point, 0), axis=1)
        else:
            # Check dominance and apply discounting
            improvement_values = []
            for s in samples:
                dominated = False
                for q in front:
                    if np.all(q >= s) and np.any(q > s):
                        dominated = True
                        break

                volume = np.prod(np.maximum(s - ref_point, 0))
                discounted_volume = volume * 0.1 if dominated else volume
                improvement_values.append(discounted_volume)

            improvement_values = np.array(improvement_values)
        
        mean_imp = np.mean(improvement_values)
        std_imp = np.std(improvement_values) 
        scores.append(mean_imp - lam * std_imp)
    
    return scores