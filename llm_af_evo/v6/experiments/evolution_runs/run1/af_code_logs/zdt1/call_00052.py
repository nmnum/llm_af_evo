def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition-score peers in feature space."""
    if len(context["pool"]) < 2:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    X_features = np.array([cand["x"] for cand in context["pool"]])
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])

    # Compute pairwise Euclidean distances between feature vectors
    diff_matrix = X_features[:, None, :] - X_features[None, :, :]
    dists = np.sqrt(np.sum(diff_matrix**2, axis=2))

    # Set diagonal to large value so candidates don't influence themselves
    np.fill_diagonal(dists, 1e6)

    # Identify high-acquisition-score peers (top 30% or at least one)
    n_high_acq = max(1, len(context["pool"]) // 3)
    top_indices = np.argpartition(-acq_values, min(n_high_acq, len(acq_values)))[:n_high_acq]
    
    # For each candidate, compute average similarity to high-acquisition peers
    avg_similarities = []
    for i in range(len(context["pool"])):
        similarities_to_peers = np.exp(-dists[i][top_indices])
        avg_similarity = np.mean(similarities_to_peers)
        avg_similarities.append(avg_similarity)

    # Convert similarity to a penalty: high similarity -> low bonus
    max_bonus = 0.3
    values = [max Bonus - (similarity * max_bonus) for similarity in avg_similarities]
    
    return values