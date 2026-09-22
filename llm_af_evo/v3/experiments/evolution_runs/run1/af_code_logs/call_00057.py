def score_pool(context):
    """Estimates Pareto optimality probability using GP posterior samples, balancing exploitation and uncertainty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Sample from each candidate's GP posteriors to estimate pareto dominance probabilities
    n_samples = 100
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples from the joint posterior distribution of objectives 
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Simple diagonal covariance matrix (independent objectives)
        cov_matrix = np.diag(stds ** 2)

        samples = np.random.multivariate_normal(means, cov_matrix, n_samples) 

        # Estimate probability of being Pareto optimal
        pf_dominance_count = sum(1 for s in samples if not any(
            all(s[i] <= pf_point[i]) and (s[i] < pf_point[i] or i == 0)
                for pf_point in context["pareto_front"]
        ) for i, _ in enumerate(names))

        # Probability of being Pareto optimal
        pareto_prob = pf_dominance_count / n_samples

        mu_sum_normed = sum(gp[name]["mean"] / front_range[name] for name in names)
        
        score = 0.5 * (mu_sum_normed + pareto_prob)

        scores.append(score)

    return scores