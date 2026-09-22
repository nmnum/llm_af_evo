def score_pool(context):
    """Rank candidates by normalized quality multiplied by inverse diversity; higher is better."""
    names = context["objective_names"]
    
    # Extract features and predicted means for each candidate
    Xs = np.array([cand['x'] for cand in context["pool"]])
    mu_sums = [sum(cand["gp_posterior"][name]["mean"] for name in names) 
               for cand in context["pool"]]
    
    # Normalize quality to be non-negative
    min_mu, max_mu = np.min(mu_sums), np.max(mu_sums)
    if max_mu - min_mu < 1e-9:
        normalized_qualities = [0.0] * len(context["pool"])
    else:
        normalized_qualities = [(mu - min_mu) / (max_mu - min_mu + 1e-9) 
                                for mu in mu_sums]
    
    # Compute pairwise similarity kernel based on feature space distance
    n_cand = Xs.shape[0]
    similarities = np.zeros((n_cand, n_cand))
    for i in range(n_cand):
        dists = np.linalg.norm(Xs - Xs[i], axis=1)
        similarities[i] = np.exp(-dists)  # similarity kernel
    
    # Compute diversity scores (inverse of average similarity excluding self)
    diversities = []
    for i in range(n_cand):
        sim_without_self = np.delete(similarities[i], i)
        avg_sim = np.mean(sim_without_self) if len(sim_without_self) > 0 else 1.0
        # Diversity is inverse: higher means less similar to others (more diverse)
        diversity_score = max(0., 1. - avg_sim) 
        diversities.append(diversity_score)

    scores = [q * d for q, d in zip(normalized_qualities, diversities)]
    
    return scores