def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable estimates."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Draw samples from the joint posterior distribution of objectives.
        means = [gp_posterior[name]["mean"] for name in context["objective_names"]]
        stds = [gp_posterior[name]["std"] for name in context["objective_names"]]
        
        # Generate correlated multivariate normal samples
        np.random.seed()  # Ensure different randomness per call if needed (if not already handled)
        draws = np.random.multivariate_normal(means, np.diag(np.square(stds)), n_samples)

        improvement_values = []
        for s in draws:
            dominated = False

            # Check dominance against the current Pareto front
            for q in context["pareto_front"]:
                if all(q[i] >= s[i] for i in range(len(s))) and any(q[i] > s[i] for i in range(len(s))):
                    dominated = True
                    break
            
            volume_to_ref_point = np.prod(np.maximum(s - ref_point, 0))
            
            # Discount the contribution of a dominated sample.
            if dominated:
                value = volume_to_ref_point * 0.1  
            else: 
                value = volume_to_ref_point

            improvement_values.append(value)
        
        mean_improvement = np.mean(improvement_values)    
        std_improvement = np.std(improvement_values)

        score = mean_improvement - lam * std_improvement
        scores.append(score)

    return scores