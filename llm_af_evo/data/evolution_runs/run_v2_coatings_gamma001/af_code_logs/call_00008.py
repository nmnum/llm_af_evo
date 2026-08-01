def score_pool(context):
    """Score candidates based on Monte Carlo sampled Pareto probability and local density of high-probability points."""
    n_samples = 100
    names = context["objective_names"]
    pf = context["pareto_front"]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Sample from the candidate's own GP posterior
        samples = np.array([[np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names] for _ in range(n_samples)])
        
        # Count how many samples are Pareto-optimal relative to current front
        n_pareto = 0
        for sample in samples:
            is_pareto = True
            for point in pf:
                if all(sample[i] <= point[i] for i in range(len(names))) and any(sample[i] < point[i] for i in range(len(names))):
                    is_pareto = False
                    break
            if is_pareto:
                n_pareto += 1
        
        pareto_prob = n_pareto / n_samples
        
        # Compute density of high-Pareto-probability points in the pool
        density = 0.0
        for other_cand in context["pool"]:
            if other_cand is cand:
                continue
            gp_other = other_cand["gp_posterior"]
            samples_other = np.array([[np.random.normal(gp_other[name]["mean"], gp_other[name]["std"]) for name in names] for _ in range(n_samples)])
            n_pareto_other = sum(1 for sample in samples_other if any(all(sample[i] <= point[i] for i in range(len(names))) and any(sample[i] < point[i] for i in range(len(names))) for point in pf))
            prob_other = n_pareto_other / n_samples
            if prob_other > 0.5:  # Only consider points with high Pareto probability
                dist = np.linalg.norm(np.array(cand["x"]) - np.array(other_cand["x"]))
                density += 1.0 / (dist + 1e-8)  # Avoid division by zero
        
        scores.append(pareto_prob * density)
    
    return scores