def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance discounting."""
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from GP posterior for this candidate
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in context["objective_names"]]
            for _ in range(n_samples)
        ])
        
        if len(front) == 0:
            # No front yet, all samples contribute full volume
            volumes = np.prod(np.maximum(samples - ref_point, 0), axis=1)
        else:
            # Check dominance and discount dominated samples
            volumes = []
            for sample in samples:
                is_dominated = False
                for point in front:
                    if np.all(point >= sample) and np.any(point > sample):
                        is_dominated = True
                        break
                
                vol = np.prod(np.maximum(sample - ref_point, 0))
                discounted_vol = vol * 0.1 if is_dominated else vol
                volumes.append(discounted_vol)
            volumes = np.array(volumes)

        mean_improvement = np.mean(volumes)
        std_improvement = np.std(volumes)
        
        score = mean_improvement - lam * std_improvement
        scores.append(score)
    
    return scores