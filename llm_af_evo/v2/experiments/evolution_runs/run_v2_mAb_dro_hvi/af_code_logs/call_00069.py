def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate predictions and scoring based on how often they dominate previously observed points."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Resample each candidate's posterior to estimate dominance probability
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample predictions from the GP posteriors (multivariate normal)
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:
                # If no uncertainty (e.g., from fixed prior or very confident GP), just use mean
                samples[:, i] = mean_val
        
        # Count how many times this candidate dominates the current Pareto front points 
        dominance_count = 0
        for sample in samples:  
            if all(sample[i] >= context["pareto_front"][:,i] for i in range(len(names))):
                # Check that at least one objective is strictly better (for true domination)
                if any(sample[i] > context["pareto_front"][:,i].max() for i in range(len(names))): 
                    dominance_count += 1

        score = float(dominance_count) / n_samples
        scores.append(score)

    return scores