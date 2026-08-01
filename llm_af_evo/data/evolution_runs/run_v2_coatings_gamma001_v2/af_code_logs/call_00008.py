def score_pool(context):
    """Estimate Pareto-optimality probability via MC sampling, then score by density of high-probability candidates."""
    n_samples = 25
    threshold = 0.5
    names = context["objective_names"]
    pf = context["pareto_front"]
    scores = []
    
    # Precompute Pareto probabilities for all candidates
    pareto_probs = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        samples = np.random.normal(
            [gp[name]["mean"] for name in names],
            [gp[name]["std"] for name in names],
            (n_samples, len(names))
        )
        n_dominated = sum(1 for s in samples if not any(
            all(s[i] >= pf_point[i] for i in range(len(names))) and
            any(s[i] > pf_point[i] for i in range(len(names)))
            for pf_point in pf
        ))
        pareto_probs.append(1.0 - n_dominated / n_samples)
    
    # Score each candidate by density of high-probability neighbors
    for i, cand in enumerate(context["pool"]):
        prob = pareto_probs[i]
        if prob < threshold:
            scores.append(0.0)
            continue
        
        x = cand["x"]
        dists = [np.linalg.norm(x - other_cand["x"]) for other_cand in context["pool"]]
        weights = np.exp(-np.array(dists) / (np.std(dists) + 1e-8))
        density = np.sum(weights * np.array(pareto_probs))
        scores.append(density)
    
    return scores