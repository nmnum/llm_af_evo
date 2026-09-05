def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with domination discounting."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    n_samples = 20
    lam = 1.0

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])

        if len(front) == 0:
            # No front yet, all samples contribute full volume
            improvement_values = [
                np.prod(np.maximum(sample - ref_point, 0))
                for sample in samples
            ]
        else:
            # Check domination and apply discounting factor of 0.1 to dominated ones.
            improvement_values = []
            for sample in samples:
                is_dominated = False

                for front_point in front:

                    if np.all(front_point >= sample) and np.any(front_point > sample):
                        is_dominated = True
                        break
                
                volume_contribution = (
                    0.1 * np.prod(np.maximum(sample - ref_point, 0))
                    if is_dominated else 
                    np.prod(np.maximum(sample - ref_point, 0)))
                
                improvement_values.append(volume_contribution)

        mean_improvement = float(np.mean(improvement_values))
        std_improvement = float(np.std(improvement_values))

        scores.append(mean_improvement - lam * std_improvement)
    
    return scores